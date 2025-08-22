#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

{r}
# app.R
library(shiny)

set.seed(123)
df <- PANSS_complete

# Define UI
ui <- fluidPage(
  titlePanel("PANSS Patient Clustering Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      # User input: number of clusters
      sliderInput("k", "Number of Clusters:",
                  min = 2, max = 8, value = 3),
      
      # Choose variables for clustering
      selectInput("xvar", "X-axis variable:",
                  choices = c("Neg","Exc","Cog","Pos","Dep","Total"), selected = "Neg"),
      selectInput("yvar", "Y-axis variable:",
                  choices = c("Neg","Exc","Cog","Pos","Dep","Total"), selected = "Pos"),
      # Filter by treatment
      selectInput("treatment", "Filter by Treatment Arm:",
                  choices = c("All", "-1 (Control)", "1 (Risperidone)"), selected = "All")
    ),
    
    mainPanel(
      plotOutput("clusterPlot"),
      tableOutput("clusterSummary")
    )
  )
)

# Define server logic
server <- function(input, output) {
  
  # Reactive filtered dataset
  filteredData <- reactive({
    if (input$treatment == "All") {
      df
    } else if (input$treatment == "-1 (Control)") {
      df %>% filter(Treat == -1)
    } else {
      df %>% filter(Treat == 1)
    }
  })
  
  # Perform clustering
  clusters <- reactive({
    df_sub <- filteredData()[, c("Neg","Exc","Cog","Pos","Dep","Total")]
    kmeans(scale(df_sub), centers = input$k, nstart = 25)
  })
  
  # Cluster plot
  output$clusterPlot <- renderPlot({
    clust_res <- clusters()
    plot_df <- filteredData() %>%
      mutate(Cluster = factor(clust_res$cluster))
    
    ggplot(plot_df, aes_string(x = input$xvar, y = input$yvar, color = "Cluster")) +
      geom_point(alpha = 0.6) +
      theme_minimal() +
      labs(title = paste("K-means Clustering with", input$k, "clusters"),
           x = input$xvar, y = input$yvar)
  })
  
  # Cluster summary stats
  output$clusterSummary <- renderTable({
    clust_res <- clusters()
    plot_df <- filteredData() %>%
      mutate(Cluster = factor(clust_res$cluster))
    
    plot_df %>%
      group_by(Cluster) %>%
      summarise(across(c(Neg, Exc, Cog, Pos, Dep, Total), mean, .names = "mean_{col}"))
  })
}

shinyApp(ui = ui, server = server)
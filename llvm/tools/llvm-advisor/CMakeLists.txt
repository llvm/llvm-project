set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(llvm-advisor
llvm-advisor.cpp
)

# Install the Python scripts alongside the binary
install(DIRECTORY view/
  DESTINATION ${CMAKE_INSTALL_BINDIR}/../share/llvm-advisor/view
  FILES_MATCHING 
  PATTERN "*.py"
  PATTERN "*.html"
  PATTERN "*.css"
  PATTERN "*.js"
  PATTERN "*.yaml"
  )

# Set executable permissions for Python scripts
install(CODE "
  execute_process(COMMAND chmod +x 
    \${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/../share/llvm-advisor/view/cli/main.py)
")

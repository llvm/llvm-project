# Gather all .cpp sources in this directory tree
file(GLOB_RECURSE LLVM_ADVISOR_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Define the executable target
add_llvm_tool(llvm-advisor
  ${LLVM_ADVISOR_SOURCES}
)

# Link required LLVM libraries
target_link_libraries(llvm-advisor PRIVATE
  LLVMSupport
  LLVMCore
  LLVMIRReader
  LLVMBitWriter
  LLVMRemarks
  LLVMProfileData
)

# Set include directories
target_include_directories(llvm-advisor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install the Python view module alongside the binary
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../view/
  DESTINATION ${CMAKE_INSTALL_BINDIR}/view
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.html"
  PATTERN "*.css"
  PATTERN "*.js"
  PATTERN "*.md"
)

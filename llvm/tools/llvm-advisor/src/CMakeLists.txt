# Gather all .cpp sources in this directory tree
file(GLOB_RECURSE LLVM_ADVISOR_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Define the executable target
add_llvm_tool(llvm-advisor
  ${LLVM_ADVISOR_SOURCES}
)

set(LLVM_ADVISOR_EXTERNAL_SOURCES
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-cov/CoverageExporterJson.cpp
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-cov/CoverageFilters.cpp
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-cov/CoverageReport.cpp
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-cov/CoverageSummaryInfo.cpp
)

target_sources(llvm-advisor PRIVATE ${LLVM_ADVISOR_EXTERNAL_SOURCES})

# Link required LLVM libraries
target_link_libraries(llvm-advisor PRIVATE
  LLVMSupport
  LLVMDebugInfoDWARF
  LLVMObject
  LLVMCore
  LLVMIRReader
  LLVMBitWriter
  LLVMRemarks
  LLVMProfileData
  clangOptions
  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangSerialization
  clangTooling
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
)

# Set include directories
set(LLVM_ADVISOR_CLANG_INCLUDE_DIRS
  ${LLVM_BINARY_DIR}/tools/clang/include
)

if(LLVM_EXTERNAL_CLANG_SOURCE_DIR)
  list(APPEND LLVM_ADVISOR_CLANG_INCLUDE_DIRS
    ${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/include
  )
endif()

target_include_directories(llvm-advisor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-cov
  ${LLVM_ADVISOR_CLANG_INCLUDE_DIRS}
)

# Install the Python webview module alongside the binary
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../tools/
  DESTINATION share/llvm-advisor/tools
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.html"
  PATTERN "*.css"
  PATTERN "*.js"
  PATTERN "*.md"
)

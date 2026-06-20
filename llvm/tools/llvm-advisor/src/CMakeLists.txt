set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  CAS
  CGData
  CodeGen
  Core
  Coverage
  DebugInfoDWARF
  Debuginfod
  IRReader
  LTO
  MCA
  MCDisassembler
  MCParser
  Object
  ProfileData
  Remarks
  Support
  Target
  XRay
)

# Gather all .cpp sources in this directory tree
file(GLOB_RECURSE LLVM_ADVISOR_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Web UI asset files that get bundled into index_html.inc
set(LLVM_ADVISOR_ASSETS
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/styles.css
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/core.js
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/shell.js
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/overview.js
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/units.js
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/unit-detail.js
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/compare.js
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/views.js
)

set(LLVM_ADVISOR_BUNDLE_INC
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/index_html.inc
)

# Regenerate bundled HTML whenever any asset source changes
add_custom_command(
  OUTPUT ${LLVM_ADVISOR_BUNDLE_INC}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/bundle.py
  DEPENDS ${LLVM_ADVISOR_ASSETS}
    ${CMAKE_CURRENT_SOURCE_DIR}/Client/HTTP/Assets/bundle.py
  COMMENT "Bundling llvm-advisor web UI assets"
  VERBATIM
)

add_custom_target(llvm-advisor-assets ALL
  DEPENDS ${LLVM_ADVISOR_BUNDLE_INC}
)

# Define the executable target
add_llvm_tool(llvm-advisor
  ${LLVM_ADVISOR_SOURCES}
)

add_dependencies(llvm-advisor llvm-advisor-assets)

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
  LLVMCAS
  LLVMDebuginfod
  LLVMXRay
  LLVMMCA
  LLVMMCDisassembler
  LLVMMCParser
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

# Install config (capability specs and profiles) alongside the binary
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../config/
  DESTINATION share/llvm-advisor/config
  FILES_MATCHING PATTERN "*.json"
)

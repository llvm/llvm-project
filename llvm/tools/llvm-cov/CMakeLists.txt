set(LLVM_LINK_COMPONENTS
  Core
  Support
  Object
  Coverage
  ProfileData
  TargetParser
  )

add_llvm_tool(llvm-cov
  llvm-cov.cpp
  gcov.cpp
  CodeCoverage.cpp
  CoverageExporterJson.cpp
  CoverageExporterLcov.cpp
  CoverageFilters.cpp
  CoverageReport.cpp
  CoverageSummaryInfo.cpp
  SourceCoverageView.cpp
  SourceCoverageViewHTML.cpp
  SourceCoverageViewText.cpp
  TestingSupport.cpp

  GENERATE_DRIVER
  )

if(LLVM_TOOL_LLVM_DRIVER_BUILD)
  set_property(GLOBAL APPEND PROPERTY LLVM_DRIVER_HIDDEN_TOOL_ALIASES_llvm-cov gcov)
endif()

target_link_libraries(llvm-cov PRIVATE LLVMDebuginfod)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  Object
  Option
  Coverage
  ProfileData
  TargetParser
  )

set(LLVM_TARGET_DEFINITIONS GcovOpts.td)
tablegen(LLVM GcovOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(CovGcovOptsTableGen)

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
  DEPENDS
  CovGcovOptsTableGen
  )

target_link_libraries(llvm-cov PRIVATE LLVMHTTP LLVMDebuginfod)

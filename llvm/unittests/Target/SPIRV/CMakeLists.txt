include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/SPIRV
  ${LLVM_BINARY_DIR}/lib/Target/SPIRV
  )

set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  BinaryFormat
  Core
  SPIRVCodeGen
  SPIRVAnalysis
  Support
  TargetParser
  )

add_llvm_target_unittest(SPIRVTests
  SPIRVConvergenceRegionAnalysisTests.cpp
  SPIRVSortBlocksTests.cpp
  SPIRVPartialOrderingVisitorTests.cpp
  SPIRVAPITest.cpp
)

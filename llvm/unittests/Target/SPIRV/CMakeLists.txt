include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/SPIRV
  ${LLVM_BINARY_DIR}/lib/Target/SPIRV
  )

set(LLVM_LINK_COMPONENTS
  AsmParser
  Core
  SPIRVCodeGen
  SPIRVAnalysis
  Support
  )

add_llvm_target_unittest(SPIRVTests
  SPIRVConvergenceRegionAnalysisTests.cpp
  )


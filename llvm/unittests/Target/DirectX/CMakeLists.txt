include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/DirectX
  ${LLVM_BINARY_DIR}/lib/Target/DirectX
  )

set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  DirectXCodeGen
  DirectXDesc
  DirectXInfo
  DirectXPointerTypeAnalysis
  MC
  Passes
  Support
  TargetParser
  )

add_llvm_target_unittest(DirectXTests
  CBufferDataLayoutTests.cpp
  PointerTypeAnalysisTests.cpp
  UniqueResourceFromUseTests.cpp
  RegisterCostTests.cpp
  )

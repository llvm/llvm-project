include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/DirectX
  ${LLVM_LIBRARY_DIR}/Target/DirectX
  )

set(LLVM_LINK_COMPONENTS
  AsmParser
  Core
  DirectXCodeGen
  Support
)

add_llvm_target_unittest(DirectXTests
  PointerTypeAnalysisTests.cpp
  )

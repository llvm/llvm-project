add_llvm_mca_unittest_includes(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_BINARY_DIR}/lib/Target/X86
  )

add_llvm_mca_unittest_sources(
  TestIncrementalMCA.cpp
  X86TestBase.cpp
  )

add_llvm_mca_unittest_link_components(
  X86
  )

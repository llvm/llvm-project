add_llvm_exegesis_unittest_includes(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV
  ${LLVM_BINARY_DIR}/lib/Target/RISCV
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

add_llvm_exegesis_unittest_link_components(
  MC
  MCParser
  Object
  Support
  Symbolize
  RISCV
  )

add_llvm_exegesis_unittest_sources(
  SnippetGeneratorTest.cpp
  TargetTest.cpp
  )
add_llvm_exegesis_unittest_link_libraries(
  LLVMExegesisRISCV)

add_llvm_exegesis_unittest_includes(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64
  ${LLVM_LIBRARY_DIR}/Target/AArch64
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

add_llvm_exegesis_unittest_link_components(
  MC
  MCParser
  Object
  Support
  Symbolize
  AArch64
  )

add_llvm_exegesis_unittest_sources(
  TargetTest.cpp
  )
add_llvm_exegesis_unittest_link_libraries(
  LLVMExegesisAArch64)

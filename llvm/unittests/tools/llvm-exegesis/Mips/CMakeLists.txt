add_llvm_exegesis_unittest_includes(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/Mips
  ${LLVM_LIBRARY_DIR}/Target/Mips
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

add_llvm_exegesis_unittest_link_components(
  MC
  MCParser
  Object
  Support
  Symbolize
  Mips
  )

add_llvm_exegesis_unittest_sources(
  BenchmarkResultTest.cpp
  RegisterAliasingTest.cpp
  SnippetGeneratorTest.cpp
  TargetTest.cpp
  )
add_llvm_exegesis_unittest_link_libraries(
  LLVMExegesisMips)

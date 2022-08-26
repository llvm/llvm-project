add_llvm_exegesis_unittest_includes(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/PowerPC
  ${LLVM_LIBRARY_DIR}/Target/PowerPC
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

add_llvm_exegesis_unittest_link_components(
  MC
  MCParser
  Object
  Support
  Symbolize
  PowerPC
  )

add_llvm_exegesis_unittest_sources(
  AnalysisTest.cpp
  SnippetGeneratorTest.cpp
  TargetTest.cpp
  )
add_llvm_exegesis_unittest_link_libraries(
  LLVMExegesisPowerPC)

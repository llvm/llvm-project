add_llvm_exegesis_unittest_includes(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_LIBRARY_DIR}/Target/X86
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

add_llvm_exegesis_unittest_sources(
  AssemblerTest.cpp
  BenchmarkResultTest.cpp
  RegisterAliasingTest.cpp
  SchedClassResolutionTest.cpp
  SnippetFileTest.cpp
  SnippetGeneratorTest.cpp
  SnippetRepetitorTest.cpp
  TargetTest.cpp
  )

add_llvm_exegesis_unittest_link_components(
  Core
  Codegen
  MC
  MCParser
  Object
  Support
  Symbolize
  X86
  )

add_llvm_exegesis_unittest_link_libraries(
  LLVMExegesisX86
  )

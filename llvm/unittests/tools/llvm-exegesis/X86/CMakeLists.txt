add_llvm_exegesis_unittest_includes(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_BINARY_DIR}/lib/Target/X86
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

add_llvm_exegesis_unittest_sources(
  BenchmarkResultTest.cpp
  RegisterAliasingTest.cpp
  SchedClassResolutionTest.cpp
  SnippetFileTest.cpp
  SnippetGeneratorTest.cpp
  SnippetRepetitorTest.cpp
  SubprocessMemoryTest.cpp
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

set(libs)
if(HAVE_LIBRT)
  list(APPEND libs rt)
endif()

add_llvm_exegesis_unittest_link_libraries(
  LLVMExegesisX86
  ${libs}
  )

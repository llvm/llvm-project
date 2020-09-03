include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_BINARY_DIR}/lib/Target/X86
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Object
  Support
  Symbolize
  X86
  )

add_llvm_target_unittest(LLVMExegesisX86Tests
  AssemblerTest.cpp
  BenchmarkResultTest.cpp
  RegisterAliasingTest.cpp
  SchedClassResolutionTest.cpp
  SnippetFileTest.cpp
  SnippetGeneratorTest.cpp
  SnippetRepetitorTest.cpp
  TargetTest.cpp
  )
target_link_libraries(LLVMExegesisX86Tests PRIVATE
  LLVMExegesis
  LLVMExegesisX86)

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/Mips
  ${LLVM_BINARY_DIR}/lib/Target/Mips
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Object
  Support
  Symbolize
  Mips
  )

add_llvm_target_unittest(LLVMExegesisMipsTests
  BenchmarkResultTest.cpp
  RegisterAliasingTest.cpp
  SnippetGeneratorTest.cpp
  TargetTest.cpp
  )
target_link_libraries(LLVMExegesisMipsTests PRIVATE
  LLVMExegesis
  LLVMExegesisMips)

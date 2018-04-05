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
  native
  )

add_llvm_unittest(LLVMExegesisTests
  BenchmarkResultTest.cpp
  InMemoryAssemblerTest.cpp
  InstructionSnippetGeneratorTest.cpp
  OperandGraphTest.cpp
  PerfHelperTest.cpp
  )
target_link_libraries(LLVMExegesisTests PRIVATE LLVMExegesis)

if(HAVE_LIBPFM)
  target_link_libraries(LLVMExegesisTests PRIVATE pfm)
endif()


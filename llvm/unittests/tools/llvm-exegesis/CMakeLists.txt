include_directories(
  ${CMAKE_SOURCE_DIR}/lib/Target/X86
  ${CMAKE_BINARY_DIR}/lib/Target/X86
  ${CMAKE_SOURCE_DIR}/tools/llvm-exegesis/lib
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


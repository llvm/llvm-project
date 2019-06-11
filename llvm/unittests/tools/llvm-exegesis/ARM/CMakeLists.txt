include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/ARM
  ${LLVM_BINARY_DIR}/lib/Target/ARM
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Object
  Support
  Symbolize
  ARM
  )

add_llvm_target_unittest(LLVMExegesisARMTests
  AssemblerTest.cpp
  )
target_link_libraries(LLVMExegesisARMTests PRIVATE LLVMExegesis)

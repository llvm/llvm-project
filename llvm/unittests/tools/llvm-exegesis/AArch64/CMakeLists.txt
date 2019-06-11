include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64
  ${LLVM_BINARY_DIR}/lib/Target/AArch64
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Object
  Support
  Symbolize
  AArch64
  )

add_llvm_target_unittest(LLVMExegesisAArch64Tests
  TargetTest.cpp
  )
target_link_libraries(LLVMExegesisAArch64Tests PRIVATE
  LLVMExegesis
  LLVMExegesisAArch64)

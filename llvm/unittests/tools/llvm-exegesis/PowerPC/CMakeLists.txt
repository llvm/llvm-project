include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/PowerPC
  ${LLVM_BINARY_DIR}/lib/Target/PowerPC
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Object
  Support
  Symbolize
  PowerPC
  )

add_llvm_unittest(LLVMExegesisPowerPCTests
  AnalysisTest.cpp
  TargetTest.cpp
  )
target_link_libraries(LLVMExegesisPowerPCTests PRIVATE
  LLVMExegesis
  LLVMExegesisPowerPC)

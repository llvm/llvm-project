add_llvm_unittest(LLVMSupportLSPTests
  Protocol.cpp
  Transport.cpp
)
target_link_libraries(LLVMSupportLSPTests
  PRIVATE
  LLVMSupportLSP)

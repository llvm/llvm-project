set(LLVM_LINK_COMPONENTS
  SupportLSP
)

add_llvm_unittest(LLVMSupportLSPTests
  Protocol.cpp
  Transport.cpp
)

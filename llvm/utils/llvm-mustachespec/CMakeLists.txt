add_llvm_utility(llvm-mustachespec
  llvm-mustachespec.cpp
  )

target_link_libraries(llvm-mustachespec PRIVATE LLVMSupport)

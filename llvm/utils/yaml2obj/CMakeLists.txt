add_llvm_utility(yaml2obj
  yaml2obj.cpp
  )

target_link_libraries(yaml2obj LLVMSupport)

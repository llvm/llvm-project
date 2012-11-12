set(LLVM_LINK_COMPONENTS archive object)

add_llvm_utility(obj2yaml
  obj2yaml.cpp coff2yaml.cpp
  )

target_link_libraries(obj2yaml LLVMSupport)

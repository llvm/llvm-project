add_llvm_utility(lit-cpuid
  lit-cpuid.cpp
  )

target_link_libraries(lit-cpuid PRIVATE LLVMSupport)

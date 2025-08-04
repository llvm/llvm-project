add_llvm_utility(llvm-test-mustache-spec
  llvm-test-mustache-spec.cpp
)

target_link_libraries(llvm-test-mustache-spec PRIVATE LLVMSupport)

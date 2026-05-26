add_llvm_utility(ExampleDaemon
    ExampleDaemon.cpp
  )

target_link_libraries(ExampleDaemon PRIVATE LLVMSupport)


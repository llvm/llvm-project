add_llvm_pass_plugin(Bye
    Bye.cpp
    DEPENDS
    intrinsics_gen
    )
if (LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(Bye PUBLIC LLVM)
else()
  target_link_libraries(Bye
    PUBLIC
    LLVMSupport
    LLVMCore
    LLVMipo
    LLVMPasses
    )
endif()


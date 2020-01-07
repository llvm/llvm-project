add_llvm_pass_plugin(Bye
    Bye.cpp
    DEPENDS
    intrinsics_gen
    BUILDTREE_ONLY
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

if( LLVM_BUILD_EXAMPLES )
  install(TARGETS ${name} RUNTIME DESTINATION examples)
endif()
set_target_properties(${name} PROPERTIES FOLDER "Examples")

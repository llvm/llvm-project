if(LLVM_BYE_LINK_INTO_TOOLS)
  message(WARNING "Setting LLVM_BYE_LINK_INTO_TOOLS=ON only makes sense for testing purpose")
endif()

add_llvm_pass_plugin(Bye
  Bye.cpp
  DEPENDS
  intrinsics_gen
  BUILDTREE_ONLY
 )

install(TARGETS ${name} RUNTIME DESTINATION examples)
set_target_properties(${name} PROPERTIES FOLDER "Examples")

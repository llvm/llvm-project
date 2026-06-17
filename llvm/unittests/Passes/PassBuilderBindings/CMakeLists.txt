set(LLVM_LINK_COMPONENTS Support Passes Core Target native AllTargetsInfos Extensions)
add_llvm_unittest(PassesBindingsTests
PassBuilderBindingsTest.cpp
)
target_link_libraries(PassesBindingsTests PRIVATE LLVMTestingSupport)
if(LLVM_BYE_LINK_INTO_TOOLS AND TARGET Bye)
  target_compile_definitions(PassesBindingsTests PRIVATE PASSBUILDER_HAS_BYE_PLUGIN=1)
endif()

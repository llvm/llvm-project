set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_unittest(LLVMProfgenTests
    ContextCompressionTest.cpp
  )

target_link_libraries(LLVMProfgenTests PRIVATE LLVMTestingSupport)

add_dependencies(LLVMProfgenTests intrinsics_gen)

set_property(TARGET LLVMProfgenTests PROPERTY FOLDER "Tests/UnitTests/ToolTests")

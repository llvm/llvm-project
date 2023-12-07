set(LLVM_LINK_COMPONENTS
  Core
  ProfileData
  Support
  )

add_llvm_unittest(LLVMProfdataTests
    OutputSizeLimitTest.cpp
    MD5CollisionTest.cpp
  )

target_link_libraries(LLVMProfdataTests PRIVATE LLVMTestingSupport)

set_property(TARGET LLVMProfdataTests PROPERTY FOLDER "Tests/UnitTests/ToolTests")

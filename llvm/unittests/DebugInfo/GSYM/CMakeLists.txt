set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  DebugInfoGSYM
  MC
  Support
  ObjectYAML
  )

add_llvm_unittest(DebugInfoGSYMTests
  GSYMTest.cpp
  )

target_link_libraries(DebugInfoGSYMTests PRIVATE LLVMTestingSupport)

set_property(TARGET DebugInfoGSYMTests PROPERTY FOLDER "Tests/UnitTests/DebugInfoTests")

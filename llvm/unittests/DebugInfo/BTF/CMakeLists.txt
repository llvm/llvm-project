set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfoBTF
  ObjectYAML
  )

add_llvm_unittest(DebugInfoBTFTests
  BTFParserTest.cpp
  )

target_link_libraries(DebugInfoBTFTests PRIVATE LLVMTestingSupport)

set_property(TARGET DebugInfoBTFTests PROPERTY FOLDER "Tests/UnitTests/DebugInfoTests")

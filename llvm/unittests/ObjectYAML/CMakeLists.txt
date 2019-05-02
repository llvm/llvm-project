set(LLVM_LINK_COMPONENTS
  Object
  ObjectYAML
  )

add_llvm_unittest(ObjectYAMLTests
  MinidumpYAMLTest.cpp
  YAMLTest.cpp
  )

target_link_libraries(ObjectYAMLTests PRIVATE LLVMTestingSupport)

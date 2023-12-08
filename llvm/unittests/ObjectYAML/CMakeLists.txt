set(LLVM_LINK_COMPONENTS
  Object
  ObjectYAML
  )

add_llvm_unittest(ObjectYAMLTests
  DWARFYAMLTest.cpp
  DXContainerYAMLTest.cpp
  ELFYAMLTest.cpp
  MinidumpYAMLTest.cpp
  YAML2ObjTest.cpp
  YAMLTest.cpp
  )

target_link_libraries(ObjectYAMLTests PRIVATE LLVMTestingSupport)

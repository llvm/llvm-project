set(LLVM_LINK_COMPONENTS
  ObjectYAML
  )

add_llvm_unittest(ObjectYAMLTests
  YAMLTest.cpp
  )


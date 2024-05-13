set(LLVM_LINK_COMPONENTS
  Support
  TestingAnnotations
  )

add_llvm_unittest(TestingAnnotationTests
  AnnotationsTest.cpp
  )

target_link_libraries(TestingAnnotationTests PRIVATE LLVMTestingAnnotations)

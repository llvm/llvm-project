set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  FrontendOpenMP
  Support
  Passes
  TransformUtils
  )

add_llvm_unittest(LLVMFrontendTests
  OpenMPContextTest.cpp
  OpenMPIRBuilderTest.cpp

  DEPENDS
  omp_gen
  )

target_link_libraries(LLVMFrontendTests PRIVATE LLVMTestingSupport)

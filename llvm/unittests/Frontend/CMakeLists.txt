set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  FrontendOpenMP
  FrontendOpenACC
  Support
  Passes
  TransformUtils
  )

add_llvm_unittest(LLVMFrontendTests
  OpenACCTest.cpp
  OpenMPContextTest.cpp
  OpenMPIRBuilderTest.cpp
  OpenMPParsingTest.cpp

  DEPENDS
  acc_gen
  omp_gen
  )

target_link_libraries(LLVMFrontendTests PRIVATE LLVMTestingSupport)

set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  FrontendOpenACC
  FrontendOpenMP
  Passes
  Support
  TargetParser
  TransformUtils
  )

add_llvm_unittest(LLVMFrontendTests
  OpenACCTest.cpp
  OpenMPContextTest.cpp
  OpenMPIRBuilderTest.cpp
  OpenMPParsingTest.cpp
  OpenMPCompositionTest.cpp
  OpenMPDecompositionTest.cpp

  DEPENDS
  acc_gen
  omp_gen
  )

target_link_libraries(LLVMFrontendTests PRIVATE LLVMTestingSupport)

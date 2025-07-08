set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  FrontendHLSL
  FrontendOpenACC
  FrontendOpenMP
  Passes
  Support
  TargetParser
  TransformUtils
  )

add_llvm_unittest(LLVMFrontendTests
  HLSLRootSignatureDumpTest.cpp
  HLSLRootSignatureRangesTest.cpp
  OpenACCTest.cpp
  OpenMPContextTest.cpp
  OpenMPIRBuilderTest.cpp
  OpenMPParsingTest.cpp
  OpenMPCompositionTest.cpp
  OpenMPDecompositionTest.cpp
  OpenMPDirectiveNameTest.cpp
  OpenMPDirectiveNameParserTest.cpp

  DEPENDS
  acc_gen
  omp_gen
  )

target_link_libraries(LLVMFrontendTests PRIVATE LLVMTestingSupport)

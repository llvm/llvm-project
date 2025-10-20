set(LLVM_LINK_COMPONENTS
  Analysis
  BinaryFormat
  Core
  FrontendHLSL
  FrontendOffloading
  FrontendOpenACC
  FrontendOpenMP
  Passes
  Support
  TargetParser
  TransformUtils
  )

add_llvm_unittest(LLVMFrontendTests
  HLSLBindingTest.cpp
  HLSLRootSignatureDumpTest.cpp
  OpenACCTest.cpp
  OpenMPContextTest.cpp
  OpenMPIRBuilderTest.cpp
  OpenMPParsingTest.cpp
  OpenMPCompositionTest.cpp
  OpenMPDecompositionTest.cpp
  OpenMPDirectiveNameTest.cpp
  OpenMPDirectiveNameParserTest.cpp
  PropertySetRegistryTest.cpp

  DEPENDS
  acc_gen
  omp_gen
  )

target_link_libraries(LLVMFrontendTests PRIVATE LLVMTestingSupport)

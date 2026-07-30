set(LLVM_LINK_COMPONENTS
  AsmParser
  Core
  FuzzMutate
  Support
  )

add_llvm_unittest(FuzzMutateTests
  OperationsTest.cpp
  ReservoirSamplerTest.cpp
  StrategiesTest.cpp
  RandomIRBuilderTest.cpp
  )

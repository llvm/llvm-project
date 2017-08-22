set(LLVM_LINK_COMPONENTS
  Core
  FuzzMutate
  Support
  )

add_llvm_unittest(FuzzMutateTests
  OperationsTest.cpp
  ReservoirSamplerTest.cpp
  )

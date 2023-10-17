set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  IPO
  Support
  TargetParser
  TransformUtils
  )

add_llvm_unittest(IPOTests
  LowerTypeTests.cpp
  WholeProgramDevirt.cpp
  AttributorTest.cpp
  FunctionSpecializationTest.cpp
  )

set_property(TARGET IPOTests PROPERTY FOLDER "Tests/UnitTests/TransformsTests")

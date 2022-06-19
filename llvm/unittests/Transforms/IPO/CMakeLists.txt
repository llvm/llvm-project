set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  IPO
  Support
  TransformUtils
  )

add_llvm_unittest(IPOTests
  LowerTypeTests.cpp
  WholeProgramDevirt.cpp
  AttributorTest.cpp
  )

set_property(TARGET IPOTests PROPERTY FOLDER "Tests/UnitTests/TransformsTests")

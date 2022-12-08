
set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Passes
  IPO
  Support
  TransformUtils
  )

add_llvm_unittest(IPOTests
  LowerTypeTests.cpp
  WholeProgramDevirt.cpp
  AttributorTest.cpp
  )

target_link_libraries(IPOTests PRIVATE LLVMTestingSupport)


set_property(TARGET IPOTests PROPERTY FOLDER "Tests/UnitTests/TransformsTests")

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
  ImportIDTableTests.cpp
  )

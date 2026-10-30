set(LLVM_LINK_COMPONENTS
  AsmParser
  SandboxIR
  Core
  Analysis
  )

add_llvm_unittest(SandboxIRTests
  IntrinsicInstTest.cpp
  PassManagerTest.cpp
  PassTest.cpp
  RegionTest.cpp
  OperatorTest.cpp
  SandboxIRTest.cpp
  TrackerTest.cpp
  TypesTest.cpp
  UtilsTest.cpp
  )

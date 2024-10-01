set(LLVM_LINK_COMPONENTS
  AsmParser
  SandboxIR
  Core
  Analysis
  )

add_llvm_unittest(SandboxIRTests
  PassTest.cpp
  SandboxIRTest.cpp
  TrackerTest.cpp
  TypesTest.cpp
  UtilsTest.cpp
  )

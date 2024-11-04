set(LLVM_LINK_COMPONENTS
  AsmParser
  SandboxIR
  Core
  )

add_llvm_unittest(SandboxIRTests
  SandboxIRTest.cpp
  TrackerTest.cpp
  TypesTest.cpp
  )

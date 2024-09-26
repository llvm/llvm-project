set(LLVM_LINK_COMPONENTS
  AsmParser
  SandboxIR
  Core
  )

add_llvm_unittest(SandboxIRTests
  PassTest.cpp
  RegionTest.cpp
  SandboxIRTest.cpp
  TrackerTest.cpp
  TypesTest.cpp
  )

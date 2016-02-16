set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  CodeGen
  Core
  MC
  MIRParser
  Support
  Target
  )

add_llvm_unittest(MITests
  LiveIntervalTest.cpp
  )

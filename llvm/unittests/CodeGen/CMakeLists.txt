set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  AsmParser
  AsmPrinter
  CodeGen
  Core
  MC
  SelectionDAG
  Support
  Target
  )

add_llvm_unittest(CodeGenTests
  AArch64SelectionDAGTest.cpp
  DIEHashTest.cpp
  LowLevelTypeTest.cpp
  MachineInstrBundleIteratorTest.cpp
  MachineInstrTest.cpp
  MachineOperandTest.cpp
  ScalableVectorMVTsTest.cpp
  )

add_subdirectory(GlobalISel)

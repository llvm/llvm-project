set(LLVM_LINK_COMPONENTS
  AsmPrinter
  CodeGen
  Core
  MC
  SelectionDAG
  Support
  Target
  )

add_llvm_unittest(CodeGenTests
  DIEHashTest.cpp
  LowLevelTypeTest.cpp
  MachineInstrBundleIteratorTest.cpp
  MachineInstrTest.cpp
  MachineOperandTest.cpp
  ScalableVectorMVTsTest.cpp
  )

add_subdirectory(GlobalISel)

set(LLVM_LINK_COMPONENTS
  AsmPrinter
  CodeGen
  Core
  MC
  SelectionDAG
  Support
  Target
  )

set(CodeGenSources
  DIEHashTest.cpp
  LowLevelTypeTest.cpp
  MachineInstrBundleIteratorTest.cpp
  MachineInstrTest.cpp
  MachineOperandTest.cpp
  ScalableVectorMVTsTest.cpp
  )

add_llvm_unittest(CodeGenTests
  ${CodeGenSources}
  )

add_subdirectory(GlobalISel)

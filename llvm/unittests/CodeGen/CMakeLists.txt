set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  AsmParser
  AsmPrinter
  CodeGen
  Core
  FileCheck
  MC
  MIRParser
  Passes
  SelectionDAG
  Support
  Target
  )

add_llvm_unittest(CodeGenTests
  AArch64SelectionDAGTest.cpp
  AllocationOrderTest.cpp
  AsmPrinterDwarfTest.cpp
  DIEHashTest.cpp
  DIETest.cpp
  InstrRefLDVTest.cpp
  LowLevelTypeTest.cpp
  LexicalScopesTest.cpp
  MachineInstrBundleIteratorTest.cpp
  MachineInstrTest.cpp
  MachineOperandTest.cpp
  PassManagerTest.cpp
  ScalableVectorMVTsTest.cpp
  SelectionDAGAddressAnalysisTest.cpp
  TypeTraitsTest.cpp
  TargetOptionsTest.cpp
  TestAsmPrinter.cpp
  )

add_subdirectory(GlobalISel)

target_link_libraries(CodeGenTests PRIVATE LLVMTestingSupport)

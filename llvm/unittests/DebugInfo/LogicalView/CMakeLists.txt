set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  DebugInfoDWARF
  DebugInfoLogicalView
  MC
  MCDisassembler
  Object
  TargetParser
  )

add_llvm_unittest_with_input_files(DebugInfoLogicalViewTests
  ../DWARF/DwarfGenerator.cpp
  ../DWARF/DwarfUtils.cpp
  CodeViewReaderTest.cpp
  CommandLineOptionsTest.cpp
  CompareElementsTest.cpp
  DWARFGeneratedTest.cpp
  DWARFReaderTest.cpp
  SelectElementsTest.cpp
  LocationRangesTest.cpp
  LogicalElementsTest.cpp
  StringPoolTest.cpp
  WarningInternalTest.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

target_link_libraries(DebugInfoLogicalViewTests PRIVATE LLVMTestingSupportStatic)

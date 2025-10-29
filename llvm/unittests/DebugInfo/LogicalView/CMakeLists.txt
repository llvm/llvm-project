set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  Core
  DebugInfoDWARF
  DebugInfoLogicalView
  MC
  MCDisassembler
  Object
  TargetParser
  TransformUtils
  )

add_llvm_unittest_with_input_files(DebugInfoLogicalViewTests
  ../DWARF/DwarfGenerator.cpp
  ../DWARF/DwarfUtils.cpp
  CodeViewReaderTest.cpp
  CommandLineOptionsTest.cpp
  CompareElementsTest.cpp
  DWARFGeneratedTest.cpp
  DWARFReaderTest.cpp
  IRReaderTest.cpp
  SelectElementsTest.cpp
  LocationRangesTest.cpp
  LogicalElementsTest.cpp
  StringPoolTest.cpp
  WarningInternalTest.cpp
  )

target_link_libraries(DebugInfoLogicalViewTests PRIVATE LLVMTestingSupport)

add_llvm_component_library(LLVMDebugInfoDWARFLowLevel
  DWARFCFIProgram.cpp
  DWARFExpression.cpp
  DWARFUnwindTable.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/DWARF/LowLevel

  # This code should have almost no dependencies. Anything that needs
  # more should use the higher-level DebugInfo/DWARF/ library.
  LINK_COMPONENTS
  BinaryFormat
  Support
  TargetParser
  )

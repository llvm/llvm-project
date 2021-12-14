add_llvm_component_library(LLVMSymbolize
  DIPrinter.cpp
  SymbolizableObjectFile.cpp
  Symbolize.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/Symbolize

  LINK_COMPONENTS
  DebugInfoDWARF
  DebugInfoPDB
  Debuginfod
  Object
  Support
  Demangle
  )

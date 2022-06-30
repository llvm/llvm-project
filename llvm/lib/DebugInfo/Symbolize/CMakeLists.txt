add_llvm_component_library(LLVMSymbolize
  DIFetcher.cpp
  DIPrinter.cpp
  Markup.cpp
  MarkupFilter.cpp
  SymbolizableObjectFile.cpp
  Symbolize.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/Symbolize

  LINK_COMPONENTS
  DebugInfoDWARF
  DebugInfoPDB
  Object
  Support
  Demangle
  )

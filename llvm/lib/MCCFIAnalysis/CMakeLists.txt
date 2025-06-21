add_llvm_component_library(LLVMMCCFIAnalysis
  FunctionUnitStreamer.cpp
  UnwindInfoHistory.cpp
  UnwindInfoAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/MCA

  LINK_COMPONENTS
  MC
  DebugInfoDWARF
  Support
  )

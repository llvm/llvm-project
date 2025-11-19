add_llvm_component_library(LLVMDWARFCFIChecker
  DWARFCFIAnalysis.cpp
  DWARFCFIFunctionFrameAnalyzer.cpp
  DWARFCFIFunctionFrameStreamer.cpp
  DWARFCFIState.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/MCA

  LINK_COMPONENTS
  MC
  DebugInfoDWARFLowLevel
  Support
  )

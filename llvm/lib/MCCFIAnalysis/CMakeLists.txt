add_llvm_component_library(LLVMMCCFIAnalysis
  CFIState.cpp
  CFIAnalysisMCStreamer.cpp
  CFIAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/MCA

  LINK_COMPONENTS
  MC
  DebugInfoDWARF
  Support
  )

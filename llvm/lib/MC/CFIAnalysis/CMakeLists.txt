add_llvm_component_library(LLVMMCCFIAnalysis
  CFIState.cpp
  # CFIAnalysisMCStreamer.cpp
  # CFIAnalysis.cpp
  # ExtendedMCInstrAnalysis.cpp

  LINK_COMPONENTS
  MC
  Support
  )

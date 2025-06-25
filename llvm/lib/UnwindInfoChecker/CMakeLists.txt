add_llvm_component_library(LLVMUnwindInfoChecker
  FunctionUnitAnalyzer.cpp
  FunctionUnitStreamer.cpp
  FunctionUnitUnwindInfoAnalyzer.cpp
  UnwindInfoAnalysis.cpp
  UnwindInfoState.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/MCA

  LINK_COMPONENTS
  MC
  DebugInfoDWARF
  Support
  )

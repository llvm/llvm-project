add_llvm_component_library(LLVMSPIRVAnalysis
  ConvergenceRegionAnalysis.cpp

  LINK_COMPONENTS
  Core
  Support

  ADD_TO_COMPONENT
  SPIRV
  )

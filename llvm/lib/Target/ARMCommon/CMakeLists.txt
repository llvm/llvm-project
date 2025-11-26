add_llvm_component_library(LLVMARMCommon
  ARMCommonInstCombineIntrinsic.cpp

  LINK_COMPONENTS
  Core
  Support
  TransformUtils
  )

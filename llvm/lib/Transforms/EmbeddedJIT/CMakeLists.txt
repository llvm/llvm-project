add_llvm_component_library(LLVMEmbeddedJIT
  EJitRegisterBitcode.cpp
  EJitRegisterPeriod.cpp
  EJitWrapperGen.cpp
  EJitPeriodHandler.cpp
  EJitAotModulePass.cpp

  LINK_COMPONENTS
  Core
  InstCombine
  ipo
  IRReader
  Support
  TransformUtils
  BitWriter
  )

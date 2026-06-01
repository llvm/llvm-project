add_llvm_component_library(LLVMEmbeddedJIT
  EJitPassOptions.cpp
  EJitRegisterBitcode.cpp
  EJitRegisterPeriod.cpp
  EJitWrapperGen.cpp
  EJitPeriodHandler.cpp
  EJitAotModulePass.cpp

  LINK_COMPONENTS
  Core
  IRReader
  Support
  TransformUtils
  BitWriter
  )

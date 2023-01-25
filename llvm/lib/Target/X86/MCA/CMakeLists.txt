add_llvm_component_library(LLVMX86TargetMCA
  X86CustomBehaviour.cpp

  LINK_COMPONENTS
  MC
  MCParser
  X86Desc
  X86Info
  Support
  TargetParser
  MCA

  ADD_TO_COMPONENT
  X86
  )

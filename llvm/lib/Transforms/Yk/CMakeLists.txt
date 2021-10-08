add_llvm_component_library(LLVMYkControlPoint
  ControlPoint.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
)

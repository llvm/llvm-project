add_llvm_component_library(LLVMToy64Desc
  Toy64InstPrinter.cpp
  Toy64MCAsmInfo.cpp
  Toy64MCTargetDesc.cpp

  DEPENDS
  Toy64CommonTableGen

  LINK_COMPONENTS
  MC
  Support
  Toy64Info

  ADD_TO_COMPONENT
  Toy64
)
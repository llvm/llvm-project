add_llvm_component_library(LLVMSC32Desc
  SC32MCTargetDesc.cpp
  SC32InstPrinter.cpp
  SC32MCAsmInfo.cpp

  LINK_COMPONENTS
  SC32Info
  Support
  MC

  ADD_TO_COMPONENT SC32
  )

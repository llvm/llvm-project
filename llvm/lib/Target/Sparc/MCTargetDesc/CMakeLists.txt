add_llvm_library(LLVMSparcDesc
  SparcMCTargetDesc.cpp
  SparcMCAsmInfo.cpp
  )

add_dependencies(LLVMSparcDesc SparcCommonTableGen)

add_llvm_library(LLVMCellSPUDesc
  SPUMCTargetDesc.cpp
  SPUMCAsmInfo.cpp
  )

add_dependencies(LLVMCellSPUDesc CellSPUCommonTableGen)

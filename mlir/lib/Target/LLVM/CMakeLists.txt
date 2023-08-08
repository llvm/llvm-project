add_mlir_library(MLIRTargetLLVM
  ModuleToObject.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVM

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  IPO
  Passes
  Support
  Target
  TargetParser
  LINK_LIBS PUBLIC
  MLIRExecutionEngineUtils
  MLIRTargetLLVMIRExport
)

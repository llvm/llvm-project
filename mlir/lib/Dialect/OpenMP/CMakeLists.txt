add_mlir_dialect_library(MLIROpenMPDialect
  IR/OpenMPDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  MLIROpenMPOpsIncGen
  MLIROpenMPOpsInterfacesIncGen
  MLIROpenMPTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  MLIROpenACCMPCommon
  )

add_mlir_dialect_library(MLIROpenMPBuiltinExternalModels
  ExternalModels/BuiltinExternalModels.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  MLIROpenMPOpsInterfacesIncGen
  MLIROpenMPTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIROpenMPDialect
)

add_mlir_dialect_library(MLIROpenMPFuncExternalModels
  ExternalModels/FuncExternalModels.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  MLIROpenMPOpsInterfacesIncGen
  MLIROpenMPTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIROpenMPDialect
  MLIRFuncDialect
)

add_mlir_dialect_library(MLIROpenMPGPUExternalModels
  ExternalModels/GPUExternalModels.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  MLIROpenMPOpsInterfacesIncGen
  MLIROpenMPTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIROpenMPDialect
  MLIRGPUDialect
)

add_mlir_dialect_library(MLIROpenMPLLVMExternalModels
  ExternalModels/LLVMExternalModels.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  MLIROpenMPOpsInterfacesIncGen
  MLIROpenMPTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIROpenMPDialect
  MLIRLLVMDialect
)

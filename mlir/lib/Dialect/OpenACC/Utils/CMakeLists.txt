add_mlir_dialect_library(MLIROpenACCUtils
  OpenACCUtils.cpp
  OpenACCUtilsCG.cpp
  OpenACCUtilsGPU.cpp
  OpenACCUtilsLoop.cpp
  OpenACCUtilsReduction.cpp
  OpenACCUtilsTiling.cpp
  OpenACCUtilsType.cpp
  OpenACCRuntimeUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenACC

  DEPENDS
  MLIROpenACCPassIncGen
  MLIROpenACCOpsIncGen
  MLIROpenACCEnumsIncGen
  MLIROpenACCAttributesIncGen
  MLIROpenACCMPOpsInterfacesIncGen
  MLIROpenACCOpsInterfacesIncGen
  MLIROpenACCTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithUtils
  MLIRComplexDialect
  MLIRDataLayoutInterfaces
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROpenACCDialect
  MLIRIR
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRSupport
  MLIRTransformUtils
)

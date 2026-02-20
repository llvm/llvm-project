add_mlir_dialect_library(MLIROpenACCUtils
  OpenACCUtils.cpp
  OpenACCUtilsCG.cpp
  OpenACCUtilsGPU.cpp
  OpenACCUtilsLoop.cpp
  OpenACCUtilsTiling.cpp

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
  MLIRDataLayoutInterfaces
  MLIRGPUDialect
  MLIROpenACCDialect
  MLIRIR
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRSupport
  MLIRTransformUtils
)

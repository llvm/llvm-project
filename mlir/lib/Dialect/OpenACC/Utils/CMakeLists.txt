add_mlir_dialect_library(MLIROpenACCUtils
  OpenACCUtilsTiling.cpp
  OpenACCUtils.cpp

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
  MLIROpenACCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_dialect_library(MLIRGENDialect
  GENDialect.cpp
  GENOps.cpp
  GENTraits.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GEN

  DEPENDS
  MLIRGENOpsIncGen
  MLIRGENOpsEnumsIncGen
  MLIRGENOpsAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

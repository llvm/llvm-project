add_mlir_dialect_library(MLIRArmSMEUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmSME/Utils

  LINK_LIBS PUBLIC
  MLIRArmSMEDialect
  MLIRDialect
  MLIRIR
  )

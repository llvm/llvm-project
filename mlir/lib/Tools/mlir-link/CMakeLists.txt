add_mlir_library(MLIRLinkLib
  MlirLinkMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-link

  LINK_LIBS PUBLIC
  MLIRFunctionInterfaces
  MLIRIR
  MLIRParser
  )

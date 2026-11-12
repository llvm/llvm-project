add_mlir_library(MLIRTranslateLib
  MlirTranslateMain.cpp
  Translation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-translate

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  )

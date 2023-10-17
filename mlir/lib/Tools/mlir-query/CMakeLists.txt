set(LLVM_LINK_COMPONENTS
  lineeditor
  )

add_mlir_library(MLIRQueryLib
  MlirQueryMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-query

  LINK_LIBS PUBLIC
  MLIRQuery
  )

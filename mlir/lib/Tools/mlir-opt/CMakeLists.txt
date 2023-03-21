add_mlir_library(MLIROptLib
  MlirOptMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-opt

  LINK_LIBS PUBLIC
  MLIRBytecodeWriter
  MLIRDebug
  MLIRObservers
  MLIRPass
  MLIRParser
  MLIRSupport
  )

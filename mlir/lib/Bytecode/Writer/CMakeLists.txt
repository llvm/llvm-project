add_mlir_library(MLIRBytecodeWriter
  BytecodeWriter.cpp
  IRNumbering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Bytecode

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRBytecodeOpInterface
  )

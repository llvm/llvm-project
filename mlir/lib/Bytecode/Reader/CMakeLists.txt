add_mlir_library(MLIRBytecodeReader
  BytecodeReader.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Bytecode

  LINK_LIBS PUBLIC
  MLIRAsmParser
  MLIRIR
  MLIRSupport
  )

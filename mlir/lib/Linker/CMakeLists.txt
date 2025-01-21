add_mlir_library(MLIRLinker
  IRMover.cpp
  Linker.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Linker

  LINK_LIBS PUBLIC
  MLIRAsmParser
  MLIRBytecodeReader
  MLIRIR
  )

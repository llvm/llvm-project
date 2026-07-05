add_subdirectory(Reader)
add_subdirectory(Writer)

add_mlir_library(MLIRBytecodeOpInterface
  BytecodeOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Bytecode

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )

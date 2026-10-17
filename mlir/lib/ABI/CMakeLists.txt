add_mlir_library(MLIRABI
  ABITypeMapper.cpp
  Targets/Test/TestTarget.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/ABI

  LINK_COMPONENTS
  ABI
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDataLayoutInterfaces
  )

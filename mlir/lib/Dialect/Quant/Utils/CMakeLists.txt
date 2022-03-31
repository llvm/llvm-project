add_mlir_dialect_library(MLIRQuantUtils
  QuantizeUtils.cpp
  UniformSupport.cpp
  FakeQuantSupport.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/QuantOps/Utils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRQuant
  MLIRSupport
  )

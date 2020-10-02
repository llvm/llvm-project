add_mlir_dialect_library(MLIRQuant
  IR/QuantOps.cpp
  IR/QuantTypes.cpp
  IR/TypeDetail.h
  IR/TypeParser.cpp
  Transforms/ConvertConst.cpp
  Transforms/ConvertSimQuant.cpp
  Utils/QuantizeUtils.cpp
  Utils/UniformSupport.cpp
  Utils/FakeQuantSupport.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/QuantOps

  DEPENDS
  MLIRQuantOpsIncGen
  MLIRQuantPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRStandard
  MLIRTransformUtils
  )

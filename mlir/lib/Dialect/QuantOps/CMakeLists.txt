add_mlir_dialect_library(MLIRQuantOps
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

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRStandardOps
  MLIRTransformUtils
  LLVMSupport
  )
add_dependencies(MLIRQuantOps
                 MLIRIR
                 MLIRPass
                 MLIRQuantOpsIncGen
                 MLIRSupport
                 MLIRStandardOps)

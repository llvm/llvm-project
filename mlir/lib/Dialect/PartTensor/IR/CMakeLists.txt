add_mlir_dialect_library(MLIRPartTensor
   PartTensorDialect.cpp

   ADDITIONAL_HEADER_DIRS
   ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/PartTensor

   DEPENDS
   MLIRPartTensorOpsIncGen
   MLIRPartTensorAttrDefsIncGen

   LINK_LIBS PUBLIC
   MLIRDialect
   MLIRIR
   MLIRDialectUtils
   MLIRIR
   MLIRInferTypeOpInterface
   MLIRSupport
   )

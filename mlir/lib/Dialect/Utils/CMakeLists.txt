add_mlir_library(MLIRDialectUtils
  IndexingUtils.cpp
  ReshapeOpsUtils.cpp
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp

  LINK_LIBS PUBLIC
  MLIRArithmeticUtils
  MLIRIR
)

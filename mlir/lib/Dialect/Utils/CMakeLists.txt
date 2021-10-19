add_mlir_library(MLIRDialectUtils
  ReshapeOpsUtils.cpp
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

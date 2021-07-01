add_mlir_library(MLIRDialectUtils
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

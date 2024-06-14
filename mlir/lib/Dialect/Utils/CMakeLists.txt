add_mlir_library(MLIRDialectUtils
  IndexingUtils.cpp
  LoopUtils.cpp
  ReshapeOpsUtils.cpp
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp

  DEPENDS
  MLIRDialectUtilsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

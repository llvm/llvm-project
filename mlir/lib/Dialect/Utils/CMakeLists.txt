add_mlir_library(MLIRDialectUtils
  IndexingUtils.cpp
  ReshapeOpsUtils.cpp
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp
  VerificationUtils.cpp

  DEPENDS
  MLIRDialectUtilsIncGen
  MLIRArithOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

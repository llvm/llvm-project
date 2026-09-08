add_mlir_library(MLIRDialectUtils
  IndexingUtils.cpp
  ReshapeOpsUtils.cpp
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp
  VerificationUtils.cpp

  DEPENDS
  MLIRDialectUtilsIncGen

  # ReshapeOpsUtils.h includes the generated Arith operation declarations.
  HEADER_LIBS
  MLIRArithDialect

  LINK_LIBS PUBLIC
  MLIRIR
)

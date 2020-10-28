add_mlir_library(MLIRRewrite
  FrozenRewritePatternList.cpp
  PatternApplicator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rewrite

  DEPENDS
  mlir-generic-headers

  LINK_LIBS PUBLIC
  MLIRIR
  )

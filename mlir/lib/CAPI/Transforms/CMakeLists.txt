add_mlir_upstream_c_api_library(MLIRCAPITransforms
  Passes.cpp
  Rewrite.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRTransformUtils
)

add_mlir_upstream_c_api_library(MLIRCAPITransforms
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRTransforms
)

add_mlir_public_c_api_library(MLIRCAPITransforms
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRTransforms
)

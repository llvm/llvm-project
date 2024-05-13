add_mlir_upstream_c_api_library(MLIRCAPIInterfaces
  Interfaces.cpp

  LINK_LIBS PUBLIC
  MLIRInferTypeOpInterface)

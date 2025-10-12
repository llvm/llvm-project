add_mlir_upstream_c_api_library(MLIRCAPIPresburger
  Presburger.cpp
  
  LINK_LIBS PUBLIC
  MLIRPresburger
  )
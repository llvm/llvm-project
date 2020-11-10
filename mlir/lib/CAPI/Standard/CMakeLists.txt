add_mlir_public_c_api_library(MLIRCAPIStandard
  StandardDialect.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRStandard
)

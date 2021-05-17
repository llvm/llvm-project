get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_mlir_public_c_api_library(MLIRCAPIConversion
  Passes.cpp

  LINK_LIBS PUBLIC
  ${conversion_libs}
)

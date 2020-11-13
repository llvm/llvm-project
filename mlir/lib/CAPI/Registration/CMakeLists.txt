# Dialect registration.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_public_c_api_library(MLIRCAPIRegistration
  Registration.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  ${dialect_libs}
)

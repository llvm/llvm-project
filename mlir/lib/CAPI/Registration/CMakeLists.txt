# Dialect registration.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_mlir_upstream_c_api_library(MLIRCAPIRegistration
  Registration.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRLLVMToLLVMIRTranslation
  ${dialect_libs}
  ${translation_libs}
  ${conversion_libs}
)

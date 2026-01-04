# Dialect registration.
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
add_mlir_upstream_c_api_library(MLIRCAPIRegisterEverything
  RegisterEverything.cpp

  LINK_LIBS PUBLIC
  ${translation_libs}

  MLIRBuiltinToLLVMIRTranslation
  MLIRCAPIIR
  MLIRCAPITransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
)

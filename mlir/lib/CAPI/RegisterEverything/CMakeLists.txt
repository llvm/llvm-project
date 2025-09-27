# Dialect registration.
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
add_mlir_upstream_c_api_library(MLIRCAPIRegisterEverything
  RegisterEverything.cpp
  PARTIAL_SOURCES_INTENDED

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

add_mlir_upstream_c_api_library(MLIRCAPIRegisterAllPasses
  RegisterAllPasses.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRRegisterAllPasses
)

add_mlir_upstream_c_api_library(MLIRCAPIRegisterAllLLVMTranslations
  RegisterAllLLVMTranslations.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  ${translation_libs}
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)

add_mlir_upstream_c_api_library(MLIRCAPIRegisterAllExtensions
  RegisterAllExtensions.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRRegisterAllExtensions
)
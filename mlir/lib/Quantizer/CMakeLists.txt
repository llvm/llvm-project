# Support.
add_mlir_library(MLIRQuantizerSupport
  Support/Configuration.cpp
  Support/ConstraintAnalysisGraph.cpp
  Support/Metadata.cpp
  Support/Statistics.cpp
  Support/TypeUtils.cpp
  Support/UniformConstraints.cpp
  Support/UniformSolvers.cpp

  ADDITIONAL_HEADER_DIRS
  )

target_link_libraries(MLIRQuantizerSupport
  PUBLIC
  MLIRIR
  MLIRQuantOps
  MLIRSupport
  MLIRStandardOps
  LLVMSupport
  )

# Configurations.
add_mlir_library(MLIRQuantizerFxpMathConfig
  Configurations/FxpMathConfig.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  MLIRFxpMathOpsIncGen
  )

target_link_libraries(MLIRQuantizerFxpMathConfig
  PUBLIC
  MLIRIR
  MLIRFxpMathOps
  MLIRQuantOps
  MLIRQuantizerSupport
  MLIRStandardOps
  MLIRSupport
  LLVMSupport
  )

# Transforms.
add_mlir_library(MLIRQuantizerTransforms
  Transforms/AddDefaultStatsTestPass.cpp
  Transforms/InferQuantizedTypesPass.cpp
  Transforms/RemoveInstrumentationPass.cpp

  ADDITIONAL_HEADER_DIRS
  )
target_link_libraries(MLIRQuantizerTransforms
  PUBLIC
  MLIRIR
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerSupport
  MLIRQuantOps
  MLIRPass
  LLVMSupport
  )

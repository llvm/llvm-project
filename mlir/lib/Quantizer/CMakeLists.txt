# Support.
add_llvm_library(MLIRQuantizerSupport
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
add_llvm_library(MLIRQuantizerFxpMathConfig
  Configurations/FxpMathConfig.cpp

  ADDITIONAL_HEADER_DIRS
  )

add_dependencies(MLIRQuantizerFxpMathConfig
  MLIRFxpMathOpsIncGen
  )
target_link_libraries(MLIRQuantizerFxpMathConfig
  PUBLIC
  MLIRIR
  MLIRFxpMathOps
  MLIRQuantOps
  MLIRQuantizerSupport
  )

# Transforms.
add_llvm_library(MLIRQuantizerTransforms
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

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

  LINK_LIBS
  MLIRIR
  MLIRQuantOps
  MLIRSupport
  MLIRStandardOps
  LLVMSupport
  )

add_dependencies(MLIRQuantizerSupport
                 MLIRIR
                 MLIRQuantOps
                 MLIRSupport
                 MLIRStandardOps)

# Configurations.
add_llvm_library(MLIRQuantizerFxpMathConfig
  Configurations/FxpMathConfig.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS
  MLIRIR
  MLIRFxpMathOps
  MLIRQuantOps
  MLIRQuantizerSupport
  MLIRStandardOps
  MLIRSupport
  LLVMSupport
  )

add_dependencies(MLIRQuantizerFxpMathConfig
                 MLIRFxpMathOpsIncGen
                 MLIRQuantizerSupport)

# Transforms.
add_llvm_library(MLIRQuantizerTransforms
  Transforms/AddDefaultStatsTestPass.cpp
  Transforms/InferQuantizedTypesPass.cpp
  Transforms/RemoveInstrumentationPass.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS
  MLIRIR
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerSupport
  MLIRQuantOps
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  LLVMSupport
  )
add_dependencies(MLIRQuantizerTransforms
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerSupport
  MLIRPass)

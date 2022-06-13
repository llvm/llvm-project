add_mlir_upstream_c_api_library(MLIRCAPIAsync
  Async.cpp
  AsyncPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRAsyncPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRAsyncDialect
  MLIRAsyncTransforms
  MLIRPass
)

add_mlir_upstream_c_api_library(MLIRCAPIControlFlow
  ControlFlow.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRControlFlowDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIGPU
  GPU.cpp
  GPUPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRGPUTransforms
  MLIRPass
)

add_mlir_upstream_c_api_library(MLIRCAPILLVM
  LLVM.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRLLVMDialect
)

add_mlir_upstream_c_api_library(MLIRCAPILinalg
  Linalg.cpp
  LinalgPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRLinalgPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRLinalgDialect
  MLIRPass
  MLIRLinalgTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPISCF
  SCF.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSCFDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIShape
  Shape.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRShapeDialect
)

add_mlir_upstream_c_api_library(MLIRCAPISparseTensor
  SparseTensor.cpp
  SparseTensorPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSparseTensorDialect
  MLIRSparseTensorTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIFunc
  Func.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRFuncDialect
)

add_mlir_upstream_c_api_library(MLIRCAPITensor
  Tensor.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTensorDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIQuant
  Quant.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRQuantDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIPDL
  PDL.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRPDLDialect
)

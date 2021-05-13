add_mlir_public_c_api_library(MLIRCAPIAsync
  Async.cpp
  AsyncPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRAsyncPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRAsync
  MLIRAsyncTransforms
  MLIRPass
)

add_mlir_public_c_api_library(MLIRCAPIGPU
  GPU.cpp
  GPUPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRGPU
  MLIRPass
)

add_mlir_public_c_api_library(MLIRCAPILinalg
  Linalg.cpp
  LinalgPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRLinalgPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRLinalg
  MLIRPass
  MLIRLinalgTransforms
)

add_mlir_public_c_api_library(MLIRCAPISCF
  SCF.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSCF
)

add_mlir_public_c_api_library(MLIRCAPIShape
  Shape.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRShape
)

add_mlir_public_c_api_library(MLIRCAPISparseTensor
  SparseTensor.cpp
  SparseTensorPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSparseTensor
  MLIRSparseTensorTransforms
)

add_mlir_public_c_api_library(MLIRCAPIStandard
  Standard.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRStandard
)

add_mlir_public_c_api_library(MLIRCAPITensor
  Tensor.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTensor
)

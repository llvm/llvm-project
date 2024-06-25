add_mlir_upstream_c_api_library(MLIRCAPIAMDGPU
  AMDGPU.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRAMDGPUDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIArith
  Arith.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRArithDialect
)

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

add_mlir_upstream_c_api_library(MLIRCAPIMath
  Math.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRMathDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIMemRef
  MemRef.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRMemRefDialect
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

add_mlir_upstream_c_api_library(MLIRCAPIIRDL
  IRDL.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRIRDL
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

add_mlir_upstream_c_api_library(MLIRCAPIMLProgram
  MLProgram.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRMLProgramDialect
)

add_mlir_upstream_c_api_library(MLIRCAPINVGPU
  NVGPU.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRNVGPUDialect
)

add_mlir_upstream_c_api_library(MLIRCAPINVVM
  NVVM.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRNVVMDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIROCDL
  ROCDL.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRROCDLDialect
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

add_mlir_upstream_c_api_library(MLIRCAPISPIRV
  SPIRV.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSPIRVDialect
)

add_mlir_upstream_c_api_library(MLIRCAPITensor
  Tensor.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTensorDialect
)

add_mlir_upstream_c_api_library(MLIRCAPITransformDialect
  Transform.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTransformDialect
)

add_mlir_upstream_c_api_library(MLIRCAPITransformDialectTransforms
  TransformInterpreter.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTransformDialectTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIQuant
  Quant.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRQuantDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIOpenMP
  OpenMP.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIROpenMPDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIPDL
  PDL.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRPDLDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIVector
  Vector.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRVectorDialect
)

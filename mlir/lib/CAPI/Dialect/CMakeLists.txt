add_mlir_upstream_c_api_library(MLIRCAPIAffine
  Affine.cpp
  AffinePasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRAffinePassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRAffineDialect
  MLIRAffineTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIAMDGPU
  AMDGPU.cpp
  AMDGPUPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRNVGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRAMDGPUDialect
  MLIRAMDGPUTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIAMX
  AMX.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRAMXDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIArith
  Arith.cpp
  ArithPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRArithDialect
  MLIRArithTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIArmNeon
  ArmNeon.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRArmNeonDialect
  MLIRArmNeonTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIArmSME
  ArmSME.cpp
  ArmSMEPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRArmSMEDialect
  MLIRArmSMETransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIArmSVE
  ArmSVE.cpp
  ArmSVEPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRArmSVEPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRArmSVEDialect
  MLIRArmSVETransforms
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

add_mlir_upstream_c_api_library(MLIRCAPIBufferization
  Bufferization.cpp
  BufferizationPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRBufferizationPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIComplex
  Complex.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR

  MLIRComplexDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIControlFlow
  ControlFlow.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRControlFlowDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIDLTI
  DLTI.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRDLTIDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIEmitC
  EmitC.cpp
  EmitCPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIREmitCDialect
  MLIREmitCTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIFunc
  Func.cpp
  FuncPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRFuncDialect
  MLIRFuncTransforms
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

add_mlir_upstream_c_api_library(MLIRCAPIIndex
  Index.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRIndexDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIIRDL
  IRDL.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRIRDL
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

add_mlir_upstream_c_api_library(MLIRCAPILLVM
  LLVM.cpp
  LLVMPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIMath
  Math.cpp
  MathPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRMathDialect
  MLIRMathTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIMemRef
  MemRef.cpp
  MemRefPasses.cpp

  PARTIAL_SOURCES_INTENDED
  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRMemRefPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRMemRefDialect
  MLIRMemRefTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIMLProgram
  MLProgram.cpp
  MLProgramPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRMLProgramPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRMLProgramDialect
  MLIRMLProgramTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIMPI
  MPI.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRMPIDialect
)

add_mlir_upstream_c_api_library(MLIRCAPINVGPU
  NVGPU.cpp
  NVGPUPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRNVGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRNVGPUDialect
  MLIRNVGPUTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPINVVM
  NVVM.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRNVVMDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIOpenACC
  OpenACC.cpp
  OpenACCPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIROpenACCPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIROpenACCDialect
  MLIROpenACCTransforms
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

add_mlir_upstream_c_api_library(MLIRCAPIPDLInterp
  PDLInterp.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRPDLInterpDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIPtr
  Ptr.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRPtrDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIQuant
  Quant.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRQuantDialect
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
  SCFPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRSCFPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSCFDialect
  MLIRSCFTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIShape
  Shape.cpp
  ShapePasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRShapeDialect
  MLIRShapeOpsTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIShard
  Shard.cpp
  ShardPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRShardPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRShardDialect
  MLIRShardTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPISMT
  SMT.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSMT
)

add_mlir_upstream_c_api_library(MLIRCAPISparseTensor
  SparseTensor.cpp
  SparseTensorPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRSparseTensorPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSparseTensorDialect
  MLIRSparseTensorTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPISPIRV
  SPIRV.cpp
  SPIRVPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRSPIRVPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSPIRVDialect
  MLIRSPIRVTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPITensor
  Tensor.cpp
  TensorPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTensorDialect
  MLIRTensorTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPITosa
  Tosa.cpp
  TosaPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRTosaPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTosaDialect
  MLIRTosaTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPITransformDialect
  Transform.cpp
  TransformPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTransformDialect
  MLIRTransformDialectTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPITransformDialectTransforms
  TransformInterpreter.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTransformDialectTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIUB
  UB.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRUBDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIVCIX
  VCIX.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRVCIXDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIVector
  Vector.cpp
  VectorPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRVectorDialect
  MLIRVectorTransforms
)

add_mlir_upstream_c_api_library(MLIRCAPIWasmSSA
  WasmSSA.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRWasmSSADialect
)

add_mlir_upstream_c_api_library(MLIRCAPIX86Vector
  X86Vector.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRX86VectorDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIXeGPU
  XeGPU.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRXeGPUDialect
)

add_mlir_upstream_c_api_library(MLIRCAPIXeVM
  XeVM.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRXeVMDialect
)

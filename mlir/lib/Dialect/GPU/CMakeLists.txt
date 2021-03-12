add_mlir_dialect_library(MLIRGPU
  IR/GPUDialect.cpp
  Transforms/AllReduceLowering.cpp
  Transforms/AsyncRegionRewriter.cpp
  Transforms/KernelOutlining.cpp
  Transforms/MemoryPromotion.cpp
  Transforms/ParallelLoopMapper.cpp
  Transforms/SerializeToBlob.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  LINK_COMPONENTS
  Core
  MC

  DEPENDS
  MLIRGPUOpsIncGen
  MLIRGPUOpInterfacesIncGen
  MLIRGPUPassIncGen
  MLIRParallelLoopMapperAttrGen
  MLIRParallelLoopMapperEnumsGen

  LINK_LIBS PUBLIC
  MLIRAsync
  MLIREDSC
  MLIRIR
  MLIRLLVMIR
  MLIRSCF
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRStandard
  MLIRSupport
  MLIRTransformUtils
  )

add_mlir_dialect_library(MLIRGPU
  IR/GPUDialect.cpp
  Transforms/AllReduceLowering.cpp
  Transforms/KernelOutlining.cpp
  Transforms/MemoryPromotion.cpp
  Transforms/ParallelLoopMapper.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  DEPENDS
  MLIRGPUOpsIncGen
  MLIRParallelLoopMapperAttrGen
  MLIRParallelLoopMapperEnumsGen
  )
target_link_libraries(MLIRGPU
  PUBLIC
  MLIREDSC
  MLIRIR
  MLIRLLVMIR
  MLIRLoopOps
  MLIRPass
  MLIRSideEffects
  MLIRStandardOps
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport
  )

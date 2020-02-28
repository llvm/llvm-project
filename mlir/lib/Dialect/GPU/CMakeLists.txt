add_mlir_dialect_library(MLIRGPU
  IR/GPUDialect.cpp
  Transforms/AllReduceLowering.cpp
  Transforms/KernelOutlining.cpp
  Transforms/MemoryPromotion.cpp
  Transforms/ParallelLoopMapper.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  LINK_LIBS
  MLIREDSC
  MLIRIR
  MLIRLLVMIR
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport
  )
add_dependencies(MLIRGPU
  MLIRGPUOpsIncGen
  )

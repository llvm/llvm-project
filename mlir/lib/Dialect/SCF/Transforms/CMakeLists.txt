set(LLVM_OPTIONAL_SOURCES
  BufferDeallocationOpInterfaceImpl.cpp
  BufferizableOpInterfaceImpl.cpp
  )

add_mlir_dialect_library(MLIRSCFTransforms
  ForallToFor.cpp
  ForallToParallel.cpp
  ForToWhile.cpp
  LoopCanonicalization.cpp
  LoopPipelining.cpp
  LoopRangeFolding.cpp
  LoopSpecialization.cpp
  ParallelForToNestedFors.cpp
  ParallelLoopCollapsing.cpp
  ParallelLoopFusion.cpp
  ParallelLoopTiling.cpp
  RotateWhileLoop.cpp
  StructuralTypeConversions.cpp
  TileUsingInterface.cpp
  WrapInZeroTripCheck.cpp
  UpliftWhileToFor.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF

  DEPENDS
  MLIRSCFPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysisBase
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRArithDialect
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  MLIRIndexDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorTransforms
  MLIRTransforms
  MLIRTransformUtils
)

add_mlir_dialect_library(MLIRSCFBufferizableOpInterfaceImpl
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRDialectUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
)

add_mlir_dialect_library(MLIRSCFBufferDeallocationOpInterfaceImpl
  BufferDeallocationOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRIR
  MLIRSCFDialect
)

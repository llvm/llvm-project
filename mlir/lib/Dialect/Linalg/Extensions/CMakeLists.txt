set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  ShardingInterfaceImpl.cpp
  )

add_mlir_extension_library(MLIRLinalgShardingInterfaceImpl
  ShardingInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg/Extensions

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRAffineDialect
  MLIRAnalysis
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRSCFDialect
  MLIRShardDialect
  MLIRShardingInterface
  MLIRShardTransforms
  MLIRTensorDialect
  )

add_mlir_extension_library(MLIRLinalgAllExtensions
  AllExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg/Extensions

  LINK_LIBS PUBLIC
  MLIRLinalgShardingInterfaceImpl
  )

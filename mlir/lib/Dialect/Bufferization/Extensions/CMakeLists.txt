set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  ShardingExtensions.cpp
  )

add_mlir_extension_library(MLIRBufferizationShardingExtensions
  ShardingExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization/Extensions

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRIR
  MLIRShardingInterface
  )

add_mlir_extension_library(MLIRBufferizationAllExtensions
  AllExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization/Extensions

  LINK_LIBS PUBLIC
  MLIRBufferizationShardingExtensions
  )

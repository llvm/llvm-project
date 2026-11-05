set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  ShardingExtensions.cpp
  )

add_mlir_extension_library(MLIRTosaShardingExtensions
  ShardingExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa/Extensions

  LINK_LIBS PUBLIC
  MLIRTosaDialect
  MLIRIR
  MLIRShardDialect
  MLIRShardingInterface
  MLIRSupport
  )

add_mlir_extension_library(MLIRTosaAllExtensions
  AllExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa/Extensions

  LINK_LIBS PUBLIC
  MLIRTosaShardingExtensions
  )

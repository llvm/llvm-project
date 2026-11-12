set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  ShardingInterfaceImpl.cpp
  )

add_mlir_extension_library(MLIRTosaShardingInterfaceImpl
  ShardingInterfaceImpl.cpp

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
  MLIRTosaShardingInterfaceImpl
  )

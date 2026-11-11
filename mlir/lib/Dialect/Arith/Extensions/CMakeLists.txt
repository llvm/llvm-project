set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  ShardingInterfaceImpl.cpp
  )

add_mlir_extension_library(MLIRArithShardingInterfaceImpl
  ShardingInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/Extensions

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRShardDialect
  MLIRShardingInterface
  )

add_mlir_extension_library(MLIRArithAllExtensions
  AllExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/Extensions

  LINK_LIBS PUBLIC
  MLIRArithShardingInterfaceImpl
  )

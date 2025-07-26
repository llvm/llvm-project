set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  InlinerExtension.cpp
  ShardingExtensions.cpp
  )

add_mlir_extension_library(MLIRFuncInlinerExtension
  InlinerExtension.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Func/Extensions

  LINK_LIBS PUBLIC
  MLIRControlFlowDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRFuncDialect
  )

add_mlir_extension_library(MLIRFuncShardingExtensions
  ShardingExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Func/Extensions

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRShardingInterface
  )


add_mlir_extension_library(MLIRFuncAllExtensions
  AllExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Func/Extensions

  LINK_LIBS PUBLIC
  MLIRFuncInlinerExtension
  MLIRFuncShardingExtensions
  )

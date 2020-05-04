# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.

set(LLVM_OPTIONAL_SOURCES
  CRunnerUtils.cpp
  ExecutionEngine.cpp
  RunnerUtils.cpp
  OptUtils.cpp
  )

add_mlir_library(MLIRExecutionEngine
  ExecutionEngine.cpp
  OptUtils.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/ExecutionEngine

  LINK_COMPONENTS
  Core
  ExecutionEngine
  Object
  OrcJIT
  JITLink
  Analysis
  AggressiveInstCombine
  InstCombine
  MC
  ScalarOpts
  Target
  Vectorize
  TransformUtils
  nativecodegen
  IPO
  )
target_link_libraries(MLIRExecutionEngine
  PUBLIC
  MLIRLLVMIR
  MLIRTargetLLVMIR
  )

add_mlir_library(mlir_c_runner_utils
  SHARED
  CRunnerUtils.cpp

  EXCLUDE_FROM_LIBMLIR
  )
set_property(TARGET mlir_c_runner_utils PROPERTY CXX_STANDARD 11)

add_mlir_library(mlir_c_runner_utils_static
  CRunnerUtils.cpp

  EXCLUDE_FROM_LIBMLIR
  )
set_property(TARGET mlir_c_runner_utils_static PROPERTY CXX_STANDARD 11)
target_compile_definitions(mlir_c_runner_utils PRIVATE mlir_c_runner_utils_EXPORTS)

add_mlir_library(mlir_runner_utils
  SHARED
  RunnerUtils.cpp

  EXCLUDE_FROM_LIBMLIR
  )
target_link_libraries(mlir_runner_utils
  PUBLIC
  mlir_c_runner_utils_static
)
target_compile_definitions(mlir_runner_utils PRIVATE mlir_runner_utils_EXPORTS)

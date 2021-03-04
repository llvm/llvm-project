# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.

set(LLVM_OPTIONAL_SOURCES
  AsyncRuntime.cpp
  CRunnerUtils.cpp
  CudaRuntimeWrappers.cpp
  SparseUtils.cpp
  ExecutionEngine.cpp
  RunnerUtils.cpp
  OptUtils.cpp
  JitRunner.cpp
  )

add_mlir_library(MLIRExecutionEngine
  ExecutionEngine.cpp
  OptUtils.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/ExecutionEngine

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Coroutines
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

  LINK_LIBS PUBLIC
  MLIRLLVMIR
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_library(MLIRJitRunner
  JitRunner.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  OrcJIT
  JITLink

  LINK_LIBS PUBLIC
  ${dialect_libs}
  MLIRExecutionEngine
  MLIRIR
  MLIRParser
  MLIRStandard
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIRStandardToLLVM
  MLIRSupport
)

add_mlir_library(mlir_c_runner_utils
  SHARED
  CRunnerUtils.cpp
  SparseUtils.cpp

  EXCLUDE_FROM_LIBMLIR
  )
set_property(TARGET mlir_c_runner_utils PROPERTY CXX_STANDARD 11)
target_compile_definitions(mlir_c_runner_utils PRIVATE mlir_c_runner_utils_EXPORTS)

add_mlir_library(mlir_runner_utils
  SHARED
  RunnerUtils.cpp

  EXCLUDE_FROM_LIBMLIR
)
target_compile_definitions(mlir_runner_utils PRIVATE mlir_runner_utils_EXPORTS)

add_mlir_library(mlir_async_runtime
  SHARED
  AsyncRuntime.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB}
)
set_property(TARGET mlir_async_runtime PROPERTY CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(mlir_async_runtime PRIVATE mlir_async_runtime_EXPORTS)

if(MLIR_CUDA_RUNNER_ENABLED)
  # Configure CUDA support. Using check_language first allows us to give a
  # custom error message.
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building the mlir cuda runner requires a working CUDA install")
  endif()

  # We need the libcuda.so library.
  find_library(CUDA_RUNTIME_LIBRARY cuda)

  add_mlir_library(mlir_cuda_runtime
    SHARED
    CudaRuntimeWrappers.cpp

    EXCLUDE_FROM_LIBMLIR
  )
  set_property(TARGET mlir_cuda_runtime PROPERTY CXX_STANDARD 14)
  target_include_directories(mlir_cuda_runtime
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(mlir_cuda_runtime
    PRIVATE
    ${CUDA_RUNTIME_LIBRARY}
  )
endif()

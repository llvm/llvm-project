# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.

set(LLVM_OPTIONAL_SOURCES
  AsyncRuntime.cpp
  CRunnerUtils.cpp
  CudaRuntimeWrappers.cpp
  SparseTensorUtils.cpp
  ExecutionEngine.cpp
  RocmRuntimeWrappers.cpp
  RunnerUtils.cpp
  OptUtils.cpp
  JitRunner.cpp
  )

if(LLVM_USE_INTEL_JITEVENTS)
  set(LLVM_JIT_LISTENER_LIB
      IntelJITEvents)
endif(LLVM_USE_INTEL_JITEVENTS)

if(LLVM_USE_PERF)
  set(LLVM_JIT_LISTENER_LIB
      PerfJITEvents)
endif(LLVM_USE_PERF)

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
  ${LLVM_JIT_LISTENER_LIB}

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
  MLIRFunc
  MLIRFuncToLLVM
  MLIRIR
  MLIRParser
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIRSupport
)

add_mlir_library(mlir_c_runner_utils
  SHARED
  CRunnerUtils.cpp
  SparseTensorUtils.cpp

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

if(MLIR_ENABLE_CUDA_RUNNER)
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

if(MLIR_ENABLE_ROCM_RUNNER)
  # Configure ROCm support.
  if (NOT DEFINED ROCM_PATH)
    if (NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
    else()
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
    endif()
    set(HIP_PATH "${ROCM_PATH}/hip" CACHE PATH "Path to which HIP has been installed")
  endif()
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP)
  if (NOT HIP_FOUND)
    message(SEND_ERROR "Building mlir with ROCm support requires a working ROCm and HIP install")
  else()
    message(STATUS "ROCm HIP version: ${HIP_VERSION}")
  endif()

  # Locate HIP runtime library.
  find_library(ROCM_RUNTIME_LIBRARY amdhip64
               PATHS "${HIP_PATH}/lib")
  if (NOT ROCM_RUNTIME_LIBRARY)
    message(SEND_ERROR "Could not locate ROCm HIP runtime library")
  else()
    message(STATUS "ROCm HIP runtime lib: ${ROCM_RUNTIME_LIBRARY}")
  endif()

  if (NOT DEFINED ROCM_TEST_CHIPSET)
    execute_process(COMMAND "${ROCM_PATH}/bin/rocm_agent_enumerator"
    OUTPUT_VARIABLE AGENTS_STRING
    ERROR_VARIABLE AGENTS_STRING
    RESULT_VARIABLE AGENT_ENUMERATOR_RESULT)

    if (NOT AGENT_ENUMERATOR_RESULT EQUAL 0)
      message(SEND_ERROR "Could not run rocm_agent_enumerator and ROCM_TEST_CHIPSET is not defined")
      set(AGENTS_STRING "")
    endif()
    string(STRIP AGENTS_STRING ${AGENTS_STRING})
    string(REPLACE "\n" ";" AGENTS_LIST ${AGENTS_STRING})
    list(FILTER AGENTS_LIST EXCLUDE REGEX "gfx000")
    if (AGENTS_LIST STREQUAL "")
      message(SEND_ERROR "No non-CPU ROCm agents found on the system, and ROCM_TEST_CHIPSET is not defined")
    else()
      list(GET AGENTS_LIST 0 FIRST_AGENT)
      set(ROCM_TEST_CHIPSET ${FIRST_AGENT} CACHE STRING "Chipset for which to compile ROCm integration tests")
      message(STATUS "Compiling integration tests for ${ROCM_TEST_CHIPSET}")
    endif()
  endif()

  add_mlir_library(mlir_rocm_runtime
    SHARED
    RocmRuntimeWrappers.cpp

    EXCLUDE_FROM_LIBMLIR
  )

  # Supress compiler warnings from HIP headers
  check_cxx_compiler_flag(-Wno-c++98-compat-extra-semi
    CXX_SUPPORTS_NO_CXX98_COMPAT_EXTRA_SEMI_FLAG)
  if (CXX_SUPPORTS_CXX98_COMPAT_EXTRA_SEMI_FLAG)
    target_compile_options(mlir_rocm_runtime PRIVATE
      "-Wno-c++98-compat-extra-semi")
  endif()
  check_cxx_compiler_flag(-Wno-return-type-c-linkage
      CXX_SUPPORTS_WNO_RETURN_TYPE_C_LINKAGE_FLAG)
  if (CXX_SUPPORTS_WNO_RETURN_TYPE_C_LINKAGE_FLAG)
    target_compile_options(mlir_rocm_runtime PRIVATE
      "-Wno-return-type-c-linkage")
  endif()
  check_cxx_compiler_flag(-Wno-nested-anon-types
    CXX_SUPPORTS_WNO_NESTED_ANON_TYPES_FLAG)
  if (CXX_SUPPORTS_WNO_NESTED_ANON_TYPES_FLAG)
    target_compile_options(mlir_rocm_runtime PRIVATE
      "-Wno-nested-anon-types")
  endif()
  check_cxx_compiler_flag(-Wno-gnu-anonymous-struct
    CXX_SUPPORTS_WNO_GNU_ANONYMOUS_STRUCT_FLAG)
  if (CXX_SUPPORTS_WNO_GNU_ANONYMOUS_STRUCT_FLAG)
    target_compile_options(mlir_rocm_runtime PRIVATE
     "-Wno-gnu-anonymous-struct")
  endif()

  target_compile_definitions(mlir_rocm_runtime
    PRIVATE
    __HIP_PLATFORM_HCC__
  )
  target_include_directories(mlir_rocm_runtime
    PRIVATE
    ${HIP_PATH}/include
    ${ROCM_PATH}/include
  )
  set_property(TARGET mlir_rocm_runtime
    PROPERTY INSTALL_RPATH_USE_LINK_PATH ON)

  target_link_libraries(mlir_rocm_runtime
    PUBLIC
    ${ROCM_RUNTIME_LIBRARY}
  )
endif()

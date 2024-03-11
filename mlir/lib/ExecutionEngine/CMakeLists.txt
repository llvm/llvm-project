# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.

set(LLVM_OPTIONAL_SOURCES
  ArmRunnerUtils.cpp
  ArmSMEStubs.cpp
  AsyncRuntime.cpp
  CRunnerUtils.cpp
  CudaRuntimeWrappers.cpp
  SparseTensorRuntime.cpp
  ExecutionEngine.cpp
  Float16bits.cpp
  RocmRuntimeWrappers.cpp
  RunnerUtils.cpp
  OptUtils.cpp
  JitRunner.cpp
  SyclRuntimeWrappers.cpp
  )

# Use a separate library for OptUtils, to avoid pulling in the entire JIT and
# codegen infrastructure. Unlike MLIRExecutionEngine, this is part of
# libMLIR.so.
add_mlir_library(MLIRExecutionEngineUtils
  OptUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/ExecutionEngine

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Coroutines
  AggressiveInstCombine
  InstCombine
  ScalarOpts
  Vectorize
  TransformUtils
  IPO
  Passes
  TargetParser
  )

if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

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
  Passes
  ${LLVM_JIT_LISTENER_LIB}

  LINK_LIBS PUBLIC
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngineUtils
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
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
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRIR
  MLIRParser
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIRSupport
)

# When -fPIC is not provided, shared libraries cannot be built if it links against
# non-PIC code.
if(LLVM_ENABLE_PIC)
  add_mlir_library(mlir_float16_utils
    SHARED
    Float16bits.cpp

    EXCLUDE_FROM_LIBMLIR
    )
  set_property(TARGET mlir_float16_utils PROPERTY CXX_STANDARD 17)
  target_compile_definitions(mlir_float16_utils PRIVATE mlir_float16_utils_EXPORTS)

  add_subdirectory(SparseTensor)

  add_mlir_library(mlir_c_runner_utils
    SHARED
    CRunnerUtils.cpp
    SparseTensorRuntime.cpp

    EXCLUDE_FROM_LIBMLIR

    LINK_LIBS PUBLIC
    mlir_float16_utils
    MLIRSparseTensorEnums
    MLIRSparseTensorRuntime
    )
  set_property(TARGET mlir_c_runner_utils PROPERTY CXX_STANDARD 17)
  target_compile_definitions(mlir_c_runner_utils PRIVATE mlir_c_runner_utils_EXPORTS)

  add_mlir_library(mlir_runner_utils
    SHARED
    RunnerUtils.cpp

    EXCLUDE_FROM_LIBMLIR

    LINK_LIBS PUBLIC
    mlir_float16_utils
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
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Don't export symbols from link-time dependencies, these are internal
    # implementation details.
    # FIXME: Add a similar fix for Windows.
    target_link_options(mlir_async_runtime PRIVATE "-Wl,-exclude-libs,ALL")
  endif()

  add_mlir_library(mlir_arm_sme_abi_stubs
    SHARED
    ArmSMEStubs.cpp)
  target_compile_definitions(mlir_arm_sme_abi_stubs PRIVATE mlir_arm_sme_abi_stubs_EXPORTS)

  add_mlir_library(mlir_arm_runner_utils
    SHARED
    ArmRunnerUtils.cpp)

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
    find_library(CUDA_RUNTIME_LIBRARY cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)

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

    if(MLIR_ENABLE_CUDA_CUSPARSE)
      # Find the libcusparse.so library if CUSPARSE build is requested.
      find_library(CUDA_CUSPARSE_LIBRARY cusparse HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)

      target_link_libraries(mlir_cuda_runtime
        PRIVATE
        ${CUDA_CUSPARSE_LIBRARY}
      )
      target_compile_definitions(mlir_cuda_runtime
        PRIVATE
        MLIR_ENABLE_CUDA_CUSPARSE=1
      )

      if(MLIR_ENABLE_CUDA_CUSPARSELT)
        # Find the libcusparseLt.so library in package manager default path if
        # CUSPARSELT build is requested. libcusparseLt.so provides sm80+ tensor
        # core support for 2:4 sparsity acceleration.
        find_library(CUDA_CUSPARSELT_LIBRARY cusparseLt HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)
        find_path(CUDA_CUSPARSELT_HEADER cusparseLt.h HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} REQUIRED)
        target_include_directories(mlir_cuda_runtime
          PRIVATE
          ${CUDA_CUSPARSELT_HEADER}
        )
        target_link_libraries(mlir_cuda_runtime
          PRIVATE
          ${CUDA_CUSPARSELT_LIBRARY}
        )
        target_compile_definitions(mlir_cuda_runtime
          PRIVATE
          MLIR_ENABLE_CUDA_CUSPARSELT=1
        )
      endif()
    endif()
  endif()

  if(MLIR_ENABLE_ROCM_RUNNER)
    # Configure ROCm support.
    if (NOT DEFINED ROCM_PATH)
      if (NOT DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
      else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
      endif()
    endif()
    # A lot of the ROCm CMake files expect to find their own dependencies in
    # CMAKE_PREFIX_PATH and don't respect PATHS or HINTS :( .
    # Therefore, temporarily add the ROCm path to CMAKE_PREFIX_PATH so we can
    # load HIP, then remove it
    set(REAL_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} "${ROCM_PATH}/hip")
    find_package(hip REQUIRED)
    set(CMAKE_PREFIX_PATH "${REAL_CMAKE_PREFIX_PATH}")

    if (NOT DEFINED ROCM_TEST_CHIPSET)
      find_program(ROCM_AGENT_ENUMERATOR rocm_agent_enumerator "${ROCM_PATH}/bin" /usr/bin /usr/local/bin)
      if(ROCM_AGENT_ENUMERATOR)
          execute_process(COMMAND "${ROCM_AGENT_ENUMERATOR}"
          OUTPUT_VARIABLE AGENTS_STRING
          ERROR_VARIABLE AGENTS_STRING
          RESULT_VARIABLE AGENT_ENUMERATOR_RESULT)
      else()
        message(SEND_ERROR "Could not find rocm_agent_enumerator")
      endif()
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

    set_property(TARGET mlir_rocm_runtime
      PROPERTY INSTALL_RPATH_USE_LINK_PATH ON)

    target_link_libraries(mlir_rocm_runtime
      PUBLIC
      hip::host hip::amdhip64
    )
  endif()

  if(MLIR_ENABLE_SYCL_RUNNER)
    find_package(SyclRuntime)

    if(NOT SyclRuntime_FOUND)
      message(FATAL_ERROR "syclRuntime not found. Please set check oneapi installation and run setvars.sh.")
    endif()

    find_package(LevelZero)

    if(NOT LevelZero_FOUND)
      message(FATAL_ERROR "LevelZero not found. Please set LEVEL_ZERO_DIR.")
    endif()

    add_mlir_library(mlir_sycl_runtime
      SHARED
      SyclRuntimeWrappers.cpp

      EXCLUDE_FROM_LIBMLIR
    )

    check_cxx_compiler_flag("-frtti" CXX_HAS_FRTTI_FLAG)
    if(NOT CXX_HAS_FRTTI_FLAG)
      message(FATAL_ERROR "CXX compiler does not accept flag -frtti")
    endif()
    target_compile_options (mlir_sycl_runtime PUBLIC -fexceptions -frtti)

    target_include_directories(mlir_sycl_runtime PRIVATE
      ${MLIR_INCLUDE_DIRS}
    )

    target_link_libraries(mlir_sycl_runtime PRIVATE LevelZero::LevelZero SyclRuntime::SyclRuntime)

    set_property(TARGET mlir_sycl_runtime APPEND PROPERTY BUILD_RPATH "${LevelZero_LIBRARIES_DIR}" "${SyclRuntime_LIBRARIES_DIR}")
  endif()
endif()

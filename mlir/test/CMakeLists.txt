add_subdirectory(CAPI)
add_subdirectory(lib)

if (MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

if (MLIR_ENABLE_SPIRV_CPU_RUNNER)
  add_subdirectory(mlir-spirv-cpu-runner)
endif()

# Provide the MLIR CMake module dir so that the out of tree Standalone
# dialect and can add it to the module path.
set(MLIR_CMAKE_DIR
  "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir")

# Passed to lit.site.cfg.py.in to set up the path where to find libraries.
set(MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

if (MLIR_INCLUDE_INTEGRATION_TESTS)
  set(INTEL_SDE_EXECUTABLE "" CACHE STRING
      "If set, arch-specific integration tests are run with Intel SDE.")
  set(ARM_EMULATOR_EXECUTABLE "" CACHE STRING
      "If set, arch-specific Arm integration tests are run with an emulator.")
  set(ARM_EMULATOR_OPTIONS "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, pass these as parameters to the emulator.")
  set(ARM_EMULATOR_LLI_EXECUTABLE "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, use this Arm native lli.")
  set(ARM_EMULATOR_UTILS_LIB_DIR "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, find Arm native utility libraries in this directory.")
  option(MLIR_RUN_AMX_TESTS "Run AMX tests.")
  option(MLIR_RUN_X86VECTOR_TESTS "Run X86Vector tests.")
  option(MLIR_RUN_CUDA_TENSOR_CORE_TESTS "Run CUDA Tensor core WMMA tests.")
  option(MLIR_RUN_ARM_SVE_TESTS "Run Arm SVE tests.")


  # The native target may not be enabled when cross compiling, raise an error.
  if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
    message(FATAL_ERROR "MLIR_INCLUDE_INTEGRATION_TESTS requires a native target")
  endif()

endif()

llvm_canonicalize_cmake_booleans(
  LLVM_BUILD_EXAMPLES
  MLIR_ENABLE_BINDINGS_PYTHON
  MLIR_ENABLE_CUDA_CONVERSIONS
  MLIR_ENABLE_CUDA_RUNNER
  MLIR_ENABLE_ROCM_CONVERSIONS
  MLIR_ENABLE_ROCM_RUNNER
  MLIR_ENABLE_SPIRV_CPU_RUNNER
  MLIR_ENABLE_VULKAN_RUNNER
  MLIR_INCLUDE_INTEGRATION_TESTS
  MLIR_RUN_AMX_TESTS
  MLIR_RUN_CUDA_TENSOR_CORE_TESTS
  MLIR_RUN_X86VECTOR_TESTS
  MLIR_RUN_ARM_SVE_TESTS
  )

set(MLIR_RUNNER_UTILS "$<TARGET_FILE:mlir_runner_utils>")
set(MLIR_C_RUNNER_UTILS "$<TARGET_FILE:mlir_c_runner_utils>")
set(MLIR_ASYNC_RUNTIME "$<TARGET_FILE:mlir_async_runtime>")

if (MLIR_ENABLE_VULKAN_RUNNER)
  set(MLIR_VULKAN_RUNTIME_WRAPPERS "$<TARGET_FILE:vulkan-runtime-wrappers>")
endif ()

if (MLIR_ENABLE_CUDA_RUNNER)
  set(MLIR_CUDA_RUNTIME "$<TARGET_FILE:mlir_cuda_runtime>")
endif ()

if (MLIR_ENABLE_ROCM_RUNNER)
  set(MLIR_ROCM_RUNTIME "$<TARGET_FILE:mlir_rocm_runtime>")
endif ()

if (MLIR_ENABLE_SPIRV_CPU_RUNNER)
  set(MLIR_TEST_SPRIV_CPU_RUNNER_C_WRAPPERS "$<TARGET_FILE:mlir_test_spirv_cpu_runner_c_wrappers>")
endif ()

# configure below runs at configure time and does not support the use of generator expressions.
# We therefore use it only to substitute the generator expressions above into the lit.site.cfg.py.in.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py.in
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )
# GENERATE runs at generation time (so at the beginning of the build) and substitutes the generator expressions
# with the actual paths of the target binaries.
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py.in)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

set(MLIR_TEST_DEPENDS
  FileCheck count not split-file
  mlir-capi-ir-test
  mlir-capi-llvm-test
  mlir-capi-pass-test
  mlir-capi-pdl-test
  mlir-capi-quant-test
  mlir-capi-sparse-tensor-test
  mlir-capi-transform-test
  mlir-linalg-ods-yaml-gen
  mlir-lsp-server
  mlir-pdll-lsp-server
  mlir-opt
  mlir-pdll
  mlir-reduce
  mlir-tblgen
  mlir-translate
  tblgen-lsp-server
  )

# The native target may not be enabled, in this case we won't
# run tests that involves executing on the host: do not build
# useless binaries.
if(LLVM_ENABLE_PIC AND TARGET ${LLVM_NATIVE_ARCH})
  list(APPEND MLIR_TEST_DEPENDS
    mlir-cpu-runner
    llc
    mlir_async_runtime
    mlir-capi-execution-engine-test
    mlir_c_runner_utils
    mlir_runner_utils
  )
endif()

if (MLIR_INCLUDE_INTEGRATION_TESTS)
  list(APPEND MLIR_TEST_DEPENDS lli)
endif()

if(MLIR_ENABLE_CUDA_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS mlir_cuda_runtime)
endif()

if(MLIR_ENABLE_ROCM_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS mlir_rocm_runtime)
endif()

list(APPEND MLIR_TEST_DEPENDS MLIRUnitTests)

if(LLVM_BUILD_EXAMPLES)
  list(APPEND MLIR_TEST_DEPENDS
    toyc-ch1
    toyc-ch2
    toyc-ch3
    toyc-ch4
    toyc-ch5
    )
  if(MLIR_ENABLE_EXECUTION_ENGINE)
    list(APPEND MLIR_TEST_DEPENDS
      toyc-ch6
      toyc-ch7
    )
  endif()
endif()

if(MLIR_ENABLE_SPIRV_CPU_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS
    mlir-spirv-cpu-runner
    mlir_test_spirv_cpu_runner_c_wrappers
  )
endif()

if(MLIR_ENABLE_VULKAN_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS
    mlir-vulkan-runner
  )
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND MLIR_TEST_DEPENDS
    MLIRPythonModules
  )
endif()

# This target can be used to just build the dependencies
# for the check-mlir target without executing the tests.
# This is useful for bots when splitting the build step
# from the testing step.
add_custom_target(check-mlir-build-only
  DEPENDS ${MLIR_TEST_DEPENDS}
)

add_lit_testsuite(check-mlir "Running the MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
  )
set_target_properties(check-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
)

include(MLIRCheckHardwareFeatures)

add_subdirectory(CAPI)
add_subdirectory(lib)

if (MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
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
  set(ARM_EMULATOR_MLIR_RUNNER_EXECUTABLE "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, use this Arm native mlir-runner.")
  set(ARM_EMULATOR_LLI_EXECUTABLE "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, use this Arm native lli.")
  set(ARM_EMULATOR_UTILS_LIB_DIR "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, find Arm native utility libraries in this directory.")
  set(MLIR_GPU_COMPILATION_TEST_FORMAT "fatbin" CACHE STRING
      "The GPU compilation format used by the tests.")
  set(ARM_SME_ABI_ROUTINES_SHLIB "" CACHE STRING
      "Path to a shared library containing Arm SME ABI routines, required for Arm SME integration tests.")
  option(MLIR_RUN_AMX_TESTS "Run AMX tests.")
  option(MLIR_RUN_X86VECTOR_TESTS "Run X86Vector tests.")
  option(MLIR_RUN_CUDA_TENSOR_CORE_TESTS "Run CUDA Tensor core WMMA tests.")
  option(MLIR_RUN_CUDA_SM80_TESTS "Run CUDA A100 tests.")
  option(MLIR_RUN_CUDA_SM80_LT_TESTS "Run CUDA A100 structured sparsity tests.")
  option(MLIR_RUN_CUDA_SM90_TESTS "Run CUDA H100 tests.")
  option(MLIR_RUN_ARM_SVE_TESTS "Run Arm SVE tests.")
  option(MLIR_RUN_ARM_SME_TESTS "Run Arm SME tests.")

  # Check whether an emulator is required - if yes then make sure that it's
  # been set.
  check_emulator(MLIR_RUN_ARM_SVE_TESTS "HWCAP_SVE" ARM_EMULATOR_EXECUTABLE)
  check_emulator(MLIR_RUN_ARM_SME_TESTS "HWCAP2_SME" ARM_EMULATOR_EXECUTABLE)

  # The native target may not be enabled when cross compiling, raise an error.
  if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
    message(FATAL_ERROR "MLIR_INCLUDE_INTEGRATION_TESTS requires a native target")
  endif()

  # When the Integration tests are requested via the MLIR_INCLUDE_INTEGRATION_TESTS
  # configuration flag, we automatically include sm80 tests when build for
  # cuSparse when the configuration flag MLIR_ENABLE_CUDA_CUSPARSE is set and
  # include sm80 lt tests when the MLIR_ENABLE_CUDA_CUSPARSELT is set in
  # addition to those.
  if (MLIR_ENABLE_CUDA_CUSPARSE)
    set(MLIR_RUN_CUDA_SM80_TESTS ON)
    if (MLIR_ENABLE_CUDA_CUSPARSELT)
      set(MLIR_RUN_CUDA_SM80_LT_TESTS ON)
    endif()
  endif()

endif()

llvm_canonicalize_cmake_booleans(
  LLVM_BUILD_EXAMPLES
  LLVM_HAS_NVPTX_TARGET
  LLVM_INCLUDE_SPIRV_TOOLS_TESTS
  MLIR_ENABLE_BINDINGS_PYTHON
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
  MLIR_RUN_ARM_SME_TESTS
  MLIR_RUN_CUDA_SM80_TESTS
  MLIR_RUN_CUDA_SM80_LT_TESTS
  MLIR_RUN_CUDA_SM90_TESTS
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

set(MLIR_TEST_DEPENDS
  mlir-capi-ir-test
  mlir-capi-irdl-test
  mlir-capi-llvm-test
  mlir-capi-pass-test
  mlir-capi-quant-test
  mlir-capi-rewrite-test
  mlir-capi-smt-test
  mlir-capi-sparse-tensor-test
  mlir-capi-transform-test
  mlir-capi-transform-interpreter-test
  mlir-capi-translation-test
  mlir-linalg-ods-yaml-gen
  mlir-lsp-server
  mlir-opt
  mlir-query
  mlir-reduce
  mlir-rewrite
  mlir-tblgen
  mlir-translate
  tblgen-lsp-server
  tblgen-to-irdl
  )
if(NOT MLIR_STANDALONE_BUILD)
  list(APPEND MLIR_TEST_DEPENDS FileCheck count not split-file yaml2obj)
endif()

set(MLIR_TEST_DEPENDS ${MLIR_TEST_DEPENDS}
  mlir-capi-pdl-test
  mlir-pdll-lsp-server
  mlir-pdll
  )

# The native target may not be enabled, in this case we won't
# run tests that involves executing on the host: do not build
# useless binaries.
if(LLVM_ENABLE_PIC AND TARGET ${LLVM_NATIVE_ARCH})
  list(APPEND MLIR_TEST_DEPENDS
    mlir-runner
    llc
    mlir_async_runtime
    mlir-capi-execution-engine-test
    mlir-capi-global-constructors-test
    mlir_c_runner_utils
    mlir_runner_utils
    mlir_float16_utils
  )
endif()

if (MLIR_INCLUDE_INTEGRATION_TESTS)
  list(APPEND MLIR_TEST_DEPENDS lli)
endif()

if(MLIR_ENABLE_CUDA_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS mlir_cuda_runtime)
endif()

if(MLIR_ENABLE_EXECUTION_ENGINE)
  list(APPEND MLIR_TEST_DEPENDS
    mlir-capi-execution-engine-test
    mlir-capi-global-constructors-test
  )
endif()

if(MLIR_ENABLE_ROCM_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS mlir_rocm_runtime)
endif()

if(MLIR_ENABLE_SYCL_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS mlir_sycl_runtime)
endif()

if(MLIR_ENABLE_LEVELZERO_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS mlir_levelzero_runtime)
endif()

if (MLIR_RUN_ARM_SME_TESTS AND NOT ARM_SME_ABI_ROUTINES_SHLIB)
  list(APPEND MLIR_TEST_DEPENDS mlir_arm_sme_abi_stubs)
endif()

if (MLIR_RUN_ARM_SVE_TESTS OR MLIR_RUN_ARM_SME_TESTS)
  list(APPEND MLIR_TEST_DEPENDS mlir_arm_runner_utils)
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
  list(APPEND MLIR_TEST_DEPENDS
    transform-opt-ch2
    transform-opt-ch3
    transform-opt-ch4
    mlir-minimal-opt
    mlir-transform-opt
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
    mlir_spirv_cpu_runtime
  )
endif()

if(MLIR_ENABLE_VULKAN_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS
    mlir_vulkan_runtime
  )
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND MLIR_TEST_DEPENDS
    MLIRPythonModules
  )
endif()

if (LLVM_INCLUDE_SPIRV_TOOLS_TESTS)
  list(APPEND MLIR_TEST_DEPENDS spirv-as)
  list(APPEND MLIR_TEST_DEPENDS spirv-val)
endif()

# This target can be used to just build the dependencies
# for the check-mlir target without executing the tests.
# This is useful for bots when splitting the build step
# from the testing step.
add_custom_target(check-mlir-build-only
  DEPENDS ${MLIR_TEST_DEPENDS}
)
set_target_properties(check-mlir-build-only PROPERTIES FOLDER "MLIR/Tests")

add_lit_testsuite(check-mlir "Running the MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
  )

add_lit_testsuites(MLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
  SKIP "^python*"
)

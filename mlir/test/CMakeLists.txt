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

# Checks whether the specified hardware capability is supported by the host
# Linux system. This is implemented by checking auxiliary vector feature
# provided by the Linux kernel.
#
# check_hwcap(
#   hwcap_spec
#   output_var
# )
#
# hwcap_spec - HWCAP value to check - these are defined in hwcap.h in the Linux
#              kernel.
#
# output_var - Output variable to use to save the results (TRUE for supported,
#              FALSE for not supported).
#
# EXAMPLES:
#
# check_hwcap("HWCAP2_SME" SME_EMULATOR_REQUIRED)
#
function(check_hwcap hwcap_spec output)
    set(hwcap_test_src
      [====[
      #include <asm/hwcap.h>
      #include <sys/auxv.h>

      int main(void)
      {
          long hwcaps = getauxval(AT_<HWCAP_VEC>);
          return (hwcaps & <HWCAP_SPEC>) != 0;
      }
      ]====]
    )

    # Extract from $hwcap_spec whether this is AT_HWCAP or AT_HWCAP2
    string(FIND ${hwcap_spec} "_" wsloc)
    string(SUBSTRING ${hwcap_spec} 0 ${wsloc} hwcap_vec)

    string(REPLACE "<HWCAP_VEC>" ${hwcap_vec} hwcap_test_src "${hwcap_test_src}")
    string(REPLACE "<HWCAP_SPEC>" ${hwcap_spec} hwcap_test_src "${hwcap_test_src}")

    set(hwcap_test_file ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hwcap_check.c)
    file(WRITE ${hwcap_test_file} "${hwcap_test_src}")

    # Compile _and_ run
    try_run(
        test_run_result test_compile_result
        ${CMAKE_BINARY_DIR}
        ${hwcap_test_file}
    )
    # Compilation will fail if hwcap_spec is not defined - this usually means
    # that your Linux kernel is too old.
    if(${test_compile_result} AND (DEFINED test_run_result))
      message(${test_run_result})
      message(STATUS "Checking whether ${hwcap_spec} is suported by the host system: ${test_run_result}")
      set(${output} ${test_run_result} PARENT_SCOPE)
    else()
      message(STATUS "Checking whether ${hwcap_spec} is suppported by the host system: FALSE")
    endif()
endfunction(check_hwcap)

# For the given group of e2e tests (defined by the `mlir_e2e_tests` flag),
# checks whether an emulator is required. If yes, verifies that the
# corresponding CMake var pointing to an emulator (`emulator_exec`) has been
# set.
#
# check_emulator(
#   mlir_e2e_tests
#   hwcap_spec
#   emulator_exec
# )
#
# EXAMPLES:
# mlir_e2e_tests  - MLIR CMake variables corresponding to the group of e2e tests
#                   to check
# hwcap_spec      - HWCAP value to check. This should correspond to the hardware
#                   capabilities required by the tests to be checked. Possible
#                   values are defined in hwcap.h in the Linux kernel.
# emulator_exec   - variable the defines the emulator (ought to be set if
#                   required, can be empty otherwise).
#
#  check_emulator(MLIR_RUN_ARM_SVE_TESTS "HWCAP_SVE" ARM_EMULATOR_EXECUTABLE)
#
function(check_emulator mlir_e2e_tests hwcap_spec emulator_exec)
  if (NOT ${mlir_e2e_tests})
    return()
  endif()

  check_hwcap(${hwcap_spec} emulator_not_required)
  if (${emulator_not_required})
    return()
  endif()

  if (${emulator_exec} STREQUAL "")
    message(FATAL_ERROR "${mlir_e2e_tests} requires an emulator, but ${emulator_exec} is not set")
  endif()

endfunction()

if (MLIR_INCLUDE_INTEGRATION_TESTS)
  set(INTEL_SDE_EXECUTABLE "" CACHE STRING
      "If set, arch-specific integration tests are run with Intel SDE.")
  set(ARM_EMULATOR_EXECUTABLE "" CACHE STRING
      "If set, arch-specific Arm integration tests are run with an emulator.")
  set(ARM_EMULATOR_OPTIONS "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, pass these as parameters to the emulator.")
  set(ARM_EMULATOR_MLIR_CPU_RUNNER_EXECUTABLE "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, use this Arm native mlir-cpu-runner.")
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
  FileCheck count not split-file
  mlir-capi-ir-test
  mlir-capi-llvm-test
  mlir-capi-pass-test
  mlir-capi-quant-test
  mlir-capi-sparse-tensor-test
  mlir-capi-transform-test
  mlir-capi-transform-interpreter-test
  mlir-capi-translation-test
  mlir-linalg-ods-yaml-gen
  mlir-lsp-server
  mlir-opt
  mlir-query
  mlir-reduce
  mlir-tblgen
  mlir-translate
  tblgen-lsp-server
  tblgen-to-irdl
  )

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
    mlir-cpu-runner
    llc
    mlir_async_runtime
    mlir-capi-execution-engine-test
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

if(MLIR_ENABLE_ROCM_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS mlir_rocm_runtime)
endif()

if(MLIR_ENABLE_SYCL_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS mlir_sycl_runtime)
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
  add_subdirectory(mlir-spirv-cpu-runner)
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

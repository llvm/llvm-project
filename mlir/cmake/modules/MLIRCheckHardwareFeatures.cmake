# A collection of helper CMake functions to detect hardware capabilities. At
# the moment these are used when configuring MLIR integration tests.

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

    set(hwcap_test_file ${CMAKE_BINARY_DIR}/temp/hwcap_check.c)
    file(WRITE ${hwcap_test_file} "${hwcap_test_src}")

    # Compile _and_ run
    try_run(
        test_run_result test_compile_result
        "${CMAKE_BINARY_DIR}"
        "${hwcap_test_file}"
    )
    # Compilation will fail if hwcap_spec is not defined - this usually means
    # that your Linux kernel is too old.
    if(${test_compile_result} AND (DEFINED test_run_result))
      message(STATUS "Checking whether ${hwcap_spec} is supported by the host system: ${test_run_result}")
      set(${output} ${test_run_result} PARENT_SCOPE)
    else()
      message(STATUS "Checking whether ${hwcap_spec} is supported by the host system: FALSE")
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
# mlir_e2e_tests  - MLIR CMake variables corresponding to the group of e2e tests
#                   to check
# hwcap_spec      - HWCAP value to check. This should correspond to the hardware
#                   capabilities required by the tests to be checked. Possible
#                   values are defined in hwcap.h in the Linux kernel.
# emulator_exec   - variable the defines the emulator (ought to be set if
#                   required, can be empty otherwise).
#
# EXAMPLES:
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

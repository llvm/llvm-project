

# Check whether the Fortran compiler already access to builtin modules.
#
# This must be wrapped in a function because
# cmake_push_check_state/cmake_pop_check_state is insufficient to isolate
# a compiler introspection environment, see
# https://gitlab.kitware.com/cmake/cmake/-/issues/27419
function (check_fortran_builtins_available)
  cmake_push_check_state(RESET)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
  check_fortran_source_compiles("
    subroutine testroutine
      use iso_c_binding
    end subroutine
    " HAVE_FORTRAN_INTRINSIC_MODS SRC_EXT F90)
  cmake_pop_check_state()
endfunction ()


# Set options to compile Fortran module files.
#
# Usage:
#
# flang_module_target(name
#   PUBLIC
#     Modules files are to be used by other Fortran sources. If a library is
#     compiled multiple times (e.g. static/shared, or msvcrt variants), only
#     one of those can be public module files; non-public modules are still
#     generated but to be forgotten deep inside the build directory to not
#     conflict with each other.
#     Also, installs the module with the toolchain.
# )
function (flang_module_target tgtname)
  set(options PUBLIC)
  cmake_parse_arguments(ARG
    "${options}"
    ""
    ""
    ${ARGN})

  if (NOT RUNTIMES_FLANG_MODULES_ENABLED)
    message(WARNING "Cannot build module files for ${tgtname} when RUNTIMES_FLANG_MODULES_ENABLED is ${RUNTIMES_FLANG_MODULES_ENABLED}")
    return ()
  endif ()

  # Let it find the other public module files
  target_compile_options(${tgtname} PRIVATE
      "$<$<COMPILE_LANGUAGE:Fortran>:-fintrinsic-modules-path=${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}>"
    )

  if (ARG_PUBLIC)
    set_target_properties(${tgtname}
      PROPERTIES
        Fortran_MODULE_DIRECTORY "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}"
      )
  else ()
    set_target_properties(${tgtname}
      PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${tgtname}.mod"
      )
  endif ()
endfunction ()
# Check whether the build environment supports building Fortran modules
# flang-rt and openmp are the only runtimes that contain Fortran modules.
#
# Sets:
#  * RUNTIMES_FLANG_MODULES_ENABLED Whether .mod files can be created
#  * CMAKE_Fortran_*                CMake Fortran toolchain info for older versions of CMake


# Check whether the Fortran compiler already has access to builtin modules. Sets
# HAVE_FORTRAN_INTRINSIC_MODS when returning.
#
# This must be wrapped in a function because
# cmake_push_check_state/cmake_pop_check_state is insufficient to isolate
# a compiler introspection environment, see
# https://gitlab.kitware.com/cmake/cmake/-/issues/27419
function (check_fortran_builtins_available)
  if (CMAKE_Fortran_COMPILER_FORCED AND CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
    # CMake's check_fortran_source_compiles/try_compile does not take a
    # user-defined CMAKE_Fortran_PREPROCESS_SOURCE into account. Instead of
    # test-compiling, ask Flang directly for the builtin module files.
    # CMAKE_Fortran_PREPROCESS_SOURCE is defined for CMake < 3.24 because it
    # does not natively recognize Flang (see below). Once we bump the required
    # CMake version, and because setting CMAKE_Fortran_PREPROCESS_SOURCE has
    # been deprecated by CMake, this workaround can be removed.
    if (NOT DEFINED FORTRAN_HAS_ISO_C_BINDING_MOD)
      message(STATUS "Performing Test ISO_C_BINDING_PATH")
      execute_process(
        COMMAND ${CMAKE_Fortran_COMPILER} ${CMAKE_Fortran_FLAGS} "-print-file-name=iso_c_binding.mod"
        OUTPUT_VARIABLE ISO_C_BINDING_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      set(FORTRAN_HAS_ISO_C_BINDING_MOD "")
      if (EXISTS "${ISO_C_BINDING_PATH}")
        message(STATUS "Performing Test ISO_C_BINDING_PATH -- Success")
        set(FORTRAN_HAS_ISO_C_BINDING_MOD TRUE CACHE INTERNAL "Existence result of ${CMAKE_Fortran_COMPILER} -print-file-name=iso_c_binding.mod")
      else ()
        message(STATUS "Performing Test ISO_C_BINDING_PATH -- Failed")
        set(FORTRAN_HAS_ISO_C_BINDING_MOD FALSE CACHE INTERNAL "Existence result of ${CMAKE_Fortran_COMPILER} -print-file-name=iso_c_binding.mod")
      endif ()
    endif ()
  else ()
    cmake_push_check_state(RESET)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
    check_fortran_source_compiles("
      subroutine testroutine
        use iso_c_binding
      end subroutine
      " FORTRAN_HAS_ISO_C_BINDING_MOD SRC_EXT F90)
    cmake_pop_check_state()
  endif ()
  set(HAVE_FORTRAN_INTRINSIC_MODS "${FORTRAN_HAS_ISO_C_BINDING_MOD}" PARENT_SCOPE)
endfunction ()


set(FORTRAN_MODULE_DEPS "")
if (CMAKE_Fortran_COMPILER AND ("flang-rt" IN_LIST LLVM_ENABLE_RUNTIMES OR "openmp" IN_LIST LLVM_ENABLE_RUNTIMES))
  cmake_path(GET CMAKE_Fortran_COMPILER STEM _Fortran_COMPILER_STEM)

  if (_Fortran_COMPILER_STEM STREQUAL "flang-new" OR _Fortran_COMPILER_STEM STREQUAL "flang")
    # CMake 3.24 is the first version of CMake that directly recognizes Flang.
    # LLVM's requirement is only CMake 3.20, teach CMake 3.20-3.23 how to use Flang, if used.
    if (CMAKE_VERSION VERSION_LESS "3.24")
      include(CMakeForceCompiler)
      CMAKE_FORCE_Fortran_COMPILER("${CMAKE_Fortran_COMPILER}" "LLVMFlang")

      set(CMAKE_Fortran_COMPILER_ID "LLVMFlang")
      set(CMAKE_Fortran_COMPILER_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")

      set(CMAKE_Fortran_SUBMODULE_SEP "-")
      set(CMAKE_Fortran_SUBMODULE_EXT ".mod")

      set(CMAKE_Fortran_PREPROCESS_SOURCE
          "<CMAKE_Fortran_COMPILER> -cpp <DEFINES> <INCLUDES> <FLAGS> -E <SOURCE> > <PREPROCESSED_SOURCE>")

      set(CMAKE_Fortran_FORMAT_FIXED_FLAG "-ffixed-form")
      set(CMAKE_Fortran_FORMAT_FREE_FLAG "-ffree-form")

      set(CMAKE_Fortran_MODDIR_FLAG "-J")

      set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_ON "-cpp")
      set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF "-nocpp")
      set(CMAKE_Fortran_POSTPROCESS_FLAG "-ffixed-line-length-72")

      set(CMAKE_Fortran_LINKER_WRAPPER_FLAG "-Wl,")
      set(CMAKE_Fortran_LINKER_WRAPPER_FLAG_SEP ",")

      set(CMAKE_Fortran_VERBOSE_FLAG "-v")

      set(CMAKE_Fortran_LINK_MODE DRIVER)
    endif ()

    # Optimization flags are only passed after CMake 3.27.4
    # https://gitlab.kitware.com/cmake/cmake/-/commit/1140087adea98bd8d8974e4c18979f4949b52c34
    if (CMAKE_VERSION VERSION_LESS "3.27.4")
      string(APPEND CMAKE_Fortran_FLAGS_DEBUG_INIT " -O0 -g")
      string(APPEND CMAKE_Fortran_FLAGS_RELWITHDEBINFO_INIT " -O2 -g")
      string(APPEND CMAKE_Fortran_FLAGS_RELEASE_INIT " -O3")
    endif ()

    # Only CMake 3.28+ pass --target= to Flang. But for cross-compiling, including
    # to nvptx amd amdgpu targets, passing the target triple is essential.
    # https://gitlab.kitware.com/cmake/cmake/-/commit/e9af7b968756e72553296ecdcde6f36606a0babf
    if (CMAKE_VERSION VERSION_LESS "3.28")
      set(CMAKE_Fortran_COMPILE_OPTIONS_TARGET "--target=")
    endif ()
  endif ()

  include(CheckFortranSourceCompiles)
  include(CheckLanguage)

  set(RUNTIMES_FLANG_MODULES_ENABLED_default OFF)
  check_language(Fortran)
  if (CMAKE_Fortran_COMPILER)
    enable_language(Fortran)

    if (CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang" AND "flang-rt" IN_LIST LLVM_ENABLE_RUNTIMES)
      # In a bootstrapping build (or any runtimes-build that includes flang-rt),
      # the intrinsic modules are not built yet. Targets can depend on
      # flang-rt-mod to ensure that flang-rt's modules are built first.
      set(FORTRAN_MODULE_DEPS flang-rt-mod)
      set(RUNTIMES_FLANG_MODULES_ENABLED_default ON)
    else ()
      # Check whether building modules works, avoid causing the entire build to
      # fail because of Fortran. The primary situation we want to support here
      # is Flang, or its intrinsic modules were built separately in a
      # non-bootstrapping build.
      check_fortran_builtins_available()
      if (HAVE_FORTRAN_INTRINSIC_MODS)
        set(RUNTIMES_FLANG_MODULES_ENABLED_default ON)
        message(STATUS "${LLVM_SUBPROJECT_TITLE}: Non-bootstrapping Fortran modules build (${CMAKE_Fortran_COMPILER_ID} located at ${CMAKE_Fortran_COMPILER})")
      else ()
        message(STATUS "Not compiling Flang modules: Not passing smoke check")
      endif ()
    endif ()
  endif ()

  option(RUNTIMES_FLANG_MODULES_ENABLED "Build Fortran modules" "${RUNTIMES_FLANG_MODULES_ENABLED_default}")
else ()
  set(RUNTIMES_FLANG_MODULES_ENABLED NO)
endif ()


# Determine the paths for Fortran .mod files.
#
# Sets:
#  * RUNTIMES_OUTPUT_RESOURCE_MOD_DIR   Path for .mod files in build dir
#  * RUNTIMES_INSTALL_RESOURCE_MOD_PATH Path for .mod files in install dir, relative to CMAKE_INSTALL_PREFIX
if (RUNTIMES_FLANG_MODULES_ENABLED)
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
    # Flang expects its builtin modules in Clang's resource directory.
    get_toolchain_module_subdir(toolchain_mod_subdir)
    extend_path(RUNTIMES_OUTPUT_RESOURCE_MOD_DIR "${RUNTIMES_OUTPUT_RESOURCE_DIR}" "${toolchain_mod_subdir}")
    extend_path(RUNTIMES_INSTALL_RESOURCE_MOD_PATH "${RUNTIMES_INSTALL_RESOURCE_PATH}" "${toolchain_mod_subdir}")
  else ()
    # For non-Flang compilers, avoid the risk of Flang accidentally picking them up.
    extend_path(RUNTIMES_OUTPUT_RESOURCE_MOD_DIR "${RUNTIMES_OUTPUT_RESOURCE_DIR}" "finclude-${CMAKE_Fortran_COMPILER_ID}")
    extend_path(RUNTIMES_INSTALL_RESOURCE_MOD_PATH "${RUNTIMES_INSTALL_RESOURCE_PATH}" "finclude-${CMAKE_Fortran_COMPILER_ID}")
  endif ()
  cmake_path(NORMAL_PATH RUNTIMES_OUTPUT_RESOURCE_MOD_DIR)
  cmake_path(NORMAL_PATH RUNTIMES_INSTALL_RESOURCE_MOD_PATH)

  # No way to find out which mod files are built by a target, so install the
  # entire output directory. flang_module_target() will copy (only) the PUBLIC
  # .mod file into the output directory.
  # https://stackoverflow.com/questions/52712416/cmake-fortran-module-directory-to-be-used-with-add-library
  set(destination "${RUNTIMES_INSTALL_RESOURCE_MOD_PATH}/..")
  cmake_path(NORMAL_PATH destination)
    install(DIRECTORY "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}"
      DESTINATION "${destination}"
    )
endif ()


# Set options to compile Fortran module files. Assumes the code above has run.
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

  target_compile_options(${tgtname} PRIVATE
      # Let non-public modules find the public module files
      "$<$<COMPILE_LANGUAGE:Fortran>:-fintrinsic-modules-path=${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}>"

      # Flang bug workaround: Reformating of cooked token buffer causes
      # identifier to be split between lines
      "$<$<COMPILE_LANGUAGE:Fortran>:SHELL:-Xflang;SHELL:-fno-reformat>"
    )

  # `flang --target=nvptx64` fails when not specifying `-march`, even when only
  # emitting .mod files. Ensure that we pass `-march`.
  if (LLVM_RUNTIMES_TARGET MATCHES "^nvptx")
    foreach (_arch IN LISTS RUNTIMES_DEVICE_ARCHITECTURES)
      target_compile_options(${tgtname} PRIVATE
        "$<$<COMPILE_LANGUAGE:Fortran>:-march=${_arch}>"
      )
    endforeach()
  endif ()

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

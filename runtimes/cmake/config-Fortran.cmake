# This file sets the following public CMake variables:
#
# RUNTIMES_ENABLE_FORTRAN - Whether support for Fortran code is available and
# enabled. This is currently not intended to be a user-configuration but
# derived from CMAKE_Fortran_COMPILER. Can also be OFF when Fortran support is
# not needed or is insufficient, e.g. if intrinsic modules are missing and
# cannot be compiled on-the-fly.
#
# RUNTIMES_FORTRAN_BUILD_DEPS - If RUNTIMES_ENABLE_FORTRAN is true, this is a
# list of dependencies that must be built before any Fortran source can be
# compiled. Contains the build targets for intrinsic modules, if necessary.
# Otherweise, it is empty.
#
# RUNTIMES_ENABLE_FLANG_MODULES - Whether to build Flang modules and emit them
# into Flang's search path. This is a CMake CACHE option defined in
# config-Fortran.cmake and default to ON iff the Fortran compiler is detected
# for be a (compatible) version of Flang. In the OFF setting, modules are still
# built, but not installed or emitted into a default path.
#
# RUNTIMES_OUTPUT_RESOURCE_MOD_DIR - Where to emit intrinsic module files in
# the build directory. Most relevant when RUNTIMES_ENABLE_FLANG_MODULES is ON.
#
# RUNTIMES_INSTALL_RESOURCE_MOD_PATH - Where to install intrinsic module files
# in the install prefix. Relative to CMAKE_INSTALL_PREFIX. Only used when
# RUNTIMES_ENABLE_FLANG_MODULES is ON.


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


set(RUNTIMES_ENABLE_FORTRAN OFF)

# Insert at least one element for
#
#    add_dependencies(target ${RUNTIMES_FORTRAN_BUILD_DEPS})
#
# to not fail
add_custom_target(fortran-dummy-dep)
set(RUNTIMES_FORTRAN_BUILD_DEPS fortran-dummy-dep)


if (CMAKE_Fortran_COMPILER)
  # Workarounds for older versions of CMake not recognizing FLang. Hence, we
  # cannot use CMAKE_Fortran_COMPILER_ID.
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
else ()
  # Do not enable Fortran support unless a Fortran compiler is passed, i.e.
  # compilation of Fortran is explicitly intended.
  # The automatically detected C/C++ and Fortran compiler may not play together.
  # An issue encountered is that CMake adds CMAKE_EXE_LINKER_FLAGS to the linker
  # line of C/C++ as well as Fortran programs, but the compiler drivers may not
  # use accept the same flags. Specifically, LLVM adds -Wl,--color-diagnostics
  # which is supported by lld, but the flag is not accepted by ld.bfd used by
  # gfortran's driver.
  return ()
endif ()


include(CheckLanguage)
check_language(Fortran)
if (CMAKE_Fortran_COMPILER)
  enable_language(Fortran)
  include(CheckFortranSourceCompiles)

  if (CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang" AND "flang-rt" IN_LIST LLVM_ENABLE_RUNTIMES)
    # In a bootstrapping build (or any runtimes-build that includes flang-rt),
    # the intrinsic modules are not built yet. Targets can depend on
    # flang-rt-mod to ensure that flang-rt's modules are built first.
    list(APPEND RUNTIMES_FORTRAN_BUILD_DEPS flang-rt-mod)
    set(RUNTIMES_ENABLE_FORTRAN ON)
    message(STATUS "Fortran support enabled using just-built Flang-RT builtin modules")
  else ()
    # Check whether building modules works, avoid causing the entire build to
    # fail because of Fortran. The primary situation we want to support here
    # is Flang, or its intrinsic modules were built separately in a
    # non-bootstrapping build.
    check_fortran_builtins_available()
    if (HAVE_FORTRAN_INTRINSIC_MODS)
      set(RUNTIMES_ENABLE_FORTRAN ON)
      message(STATUS "Fortran support enabled using compiler's own modules")
    else ()
      message(STATUS "Fortran support disabled: Not passing smoke check")
    endif ()
  endif ()
else ()
  message(STATUS "Fortran support disabled: not enabled in CMake; Use CMAKE_Fortran_COMPILER_WORKS=yes if the issues is missing builtin modules")
endif ()


# Check whether modules files are compatible with our version of Flang.
set(RUNTIMES_ENABLE_FLANG_MODULES_default OFF)
if (CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
  set(RUNTIMES_ENABLE_FLANG_MODULES_default ON)
else ()
  set(RUNTIMES_ENABLE_FLANG_MODULES_default OFF)
endif ()
option(RUNTIMES_ENABLE_FLANG_MODULES "Make Fortran .mod files available to Flang; should only be enabled if compiling with a matching version of Flang" "${RUNTIMES_ENABLE_FLANG_MODULES_default}")


# Determine the paths for Fortran .mod files.
if (RUNTIMES_ENABLE_FLANG_MODULES)
  # Flang expects its builtin modules in Clang's resource directory.
  get_toolchain_module_subdir(toolchain_mod_subdir)
  extend_path(RUNTIMES_OUTPUT_RESOURCE_MOD_DIR "${RUNTIMES_OUTPUT_RESOURCE_DIR}" "${toolchain_mod_subdir}")
  extend_path(RUNTIMES_INSTALL_RESOURCE_MOD_PATH "${RUNTIMES_INSTALL_RESOURCE_PATH}" "${toolchain_mod_subdir}")
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

  # The INSTALL'ed directory must exist, even if empty, or `ninja install` will
  # fail with an error.
  file(MAKE_DIRECTORY "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}")
else ()
  # If Flang modules are disabled (e.g. because the compiler is not Flang), avoid the risk of Flang accidentally picking them up.
  extend_path(RUNTIMES_OUTPUT_RESOURCE_MOD_DIR "${CMAKE_CURRENT_BINARY_DIR}" "finclude-${CMAKE_Fortran_COMPILER_ID}")

  # We don't know how to install modules for other compilers. Do not install them at all.
  set(RUNTIMES_INSTALL_RESOURCE_MOD_PATH "")
endif ()
cmake_path(NORMAL_PATH RUNTIMES_OUTPUT_RESOURCE_MOD_DIR)


# Set options to compile Fortran module files. Assumes the code above has run.
#
# Usage:
#
# flang_module_target(name
#   PUBLIC
#     Modules files are to be used by other Fortran sources. If a library is
#     compiled multiple times (e.g. static/shared, or msvcrt variants), only
#     one of those can be public module files; non-public modules are still
#     generated but to be forgotten inside the build directory to not
#     conflict with each other.
#     Also, installs the module with the toolchain.
# )
function (flang_module_target tgtname)
  set(options PUBLIC)
  cmake_parse_arguments(ARG
    "${options}"
    ""
    ""
    ${ARGN}
  )

  # Let all modules find the public module files
  target_compile_options(${tgtname} PRIVATE
    "$<$<COMPILE_LANGUAGE:Fortran>:-fintrinsic-modules-path=${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}>"
  )

  if (CMAKE_Fortran_COMPILER_ID MATCHES "LLVM")
    target_compile_options(${tgtname} PRIVATE
      # Flang bug workaround: Reformating of cooked token buffer causes
      # identifier to be split between lines
      "$<$<COMPILE_LANGUAGE:Fortran>:SHELL:-Xflang;SHELL:-fno-reformat>"
    )
  endif ()

  if (ARG_PUBLIC)
    set_target_properties(${tgtname}
      PROPERTIES
        Fortran_MODULE_DIRECTORY "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}"
    )
  else ()
    # Keep non-public modules where CMake would put them normally;
    # Modules of different target must not overwrite each other.
  endif ()
endfunction ()

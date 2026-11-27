# This file sets the following public CMake variables:
#
# RUNTIMES_ENABLE_FORTRAN - Whether support for Fortran code is available and
# enabled. This is currently not intended to be a user-configuration but
# derived from CMAKE_Fortran_COMPILER. Can also be OFF when Fortran support is
# not needed or is insufficient, e.g. if intrinsic modules are missing and
# cannot be compiled on-the-fly.
#
# RUNTIMES_FORTRAN_MODULES - Whether to build Flang modules and emit them
# into Flang's search path. This is a CMake CACHE option defined in
# config-Fortran.cmake and default to ON iff the Fortran compiler is detected
# for be a (compatible) version of Flang. In the OFF setting, modules are still
# built, but not installed or emitted into a default path.
#
# RUNTIMES_OUTPUT_RESOURCE_MOD_DIR - Where to emit intrinsic module files in
# the build directory. Most relevant when RUNTIMES_FORTRAN_MODULES is ON.
#
# RUNTIMES_INSTALL_RESOURCE_MOD_PATH - Where to install intrinsic module files
# in the install prefix. Relative to CMAKE_INSTALL_PREFIX. Only used when
# RUNTIMES_FORTRAN_MODULES is ON.
#
#
# The following targets are defined:
#
# fortran-compile-options - sets the compile options necessary to compile
# modules. Primarily sets up the intrinsic module search paths if necessary.
# Gets applied to flang-rt as well.
#
# fortran-compile-depends - link to this INTERFACE library to ensure Fortran
# sources can be compiled. Ensures that the builtin modules from
# flang-rt are built before.


# Check whether the Fortran compiler already has access to builtin modules. Sets
# HAVE_FORTRAN_INTRINSIC_MODS when returning.
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
    " FORTRAN_HAS_ISO_C_BINDING_MOD SRC_EXT F90)
  cmake_pop_check_state()
  set(HAVE_FORTRAN_INTRINSIC_MODS "${FORTRAN_HAS_ISO_C_BINDING_MOD}" PARENT_SCOPE)
endfunction ()


set(RUNTIMES_ENABLE_FORTRAN OFF)

# Targets compiling Fortran sources can depend on this target to ensure toolchain
# prerequisites (runtime library and intrinsic modules) are built before.
add_library(fortran-compile-options INTERFACE)
add_library(fortran-compile-depends INTERFACE)
target_link_libraries(fortran-compile-depends INTERFACE fortran-compile-options)

if (NOT CMAKE_Fortran_COMPILER)
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

  if (CMAKE_Fortran_COMPILER_ID MATCHES "LLVM")
    target_compile_options(fortran-compile-options INTERFACE
      # Flang bug workaround: Reformatting of cooked token buffer causes
      # identifier to be split between lines
      "$<$<COMPILE_LANGUAGE:Fortran>:SHELL:-Xflang;SHELL:-fno-reformat>"
    )
  endif ()

  if (CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang" AND "flang-rt" IN_LIST LLVM_ENABLE_RUNTIMES)
    # In a bootstrapping build (or any runtimes-build that includes flang-rt),
    # the intrinsic modules are not built yet. Targets can depend on
    # flang-rt-mod to ensure that flang-rt's modules are built first.
    target_link_libraries(fortran-compile-depends INTERFACE flang-rt-mod flang-rt-mod.barrier)
    set(RUNTIMES_ENABLE_FORTRAN ON)
    message(STATUS "Fortran support enabled using just-built Flang-RT builtin modules")
  elseif (LLVM_TREE_AVAILABLE)
    # Bootstrapping build without flang-rt: CMAKE_Fortran_COMPILER is
    # necessarily Flang, but must avoid compiler introspection such as
    # check_fortran_builtins_available() below. Because flang-rt is needed for
    # building other Fortran code, disable Fortran.
    message(STATUS "Fortran support disabled: bootstrapping build without LLVM_ENABLE_RUNTIMES=flang-rt")
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
set(RUNTIMES_FORTRAN_MODULES_default OFF)
if (RUNTIMES_ENABLE_FORTRAN AND CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
  set(RUNTIMES_FORTRAN_MODULES_default ON)
endif ()
option(RUNTIMES_FORTRAN_MODULES "Make Fortran .mod files available to Flang; should only be enabled if compiling with a matching version of Flang" "${RUNTIMES_FORTRAN_MODULES_default}")

# Determine the paths for Fortran .mod files.
if (RUNTIMES_FORTRAN_MODULES)
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

  # Let all Fortran sources find the public module files
  target_compile_options(fortran-compile-options INTERFACE
    "$<$<COMPILE_LANGUAGE:Fortran>:-fintrinsic-modules-path=${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}>"
  )
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

  if (tgtname STREQUAL "flang-rt-mod")
    # Avoid circular dependency on flang-rt itself
    target_link_libraries(${tgtname} PRIVATE fortran-compile-options)
  else ()
    target_link_libraries(${tgtname} PRIVATE fortran-compile-depends)
  endif ()

  # Make CMake not ignore "use, intrinsic ::"-dependencies. Unfortunately,
  # it is not universally handled by CMake s.t. we currently must not use
  # "use, intrinsic ::" in our sources.
  #  * CMake 3.22: Added Fortran_BUILDING_INSTRINSIC_MODULES handling "Unix Makefiles" generator
  #  * CMake 4.0: Renamed INSTRINSIC to INTRINSIC (typo fix); INSTRINSIC spelling kept but deprecated
  #  * CMake 4.5: Added handling by Ninja generators as well
  set_target_properties(${tgtname}
    PROPERTIES
      Fortran_BUILDING_INTRINSIC_MODULES ON
      Fortran_BUILDING_INSTRINSIC_MODULES ON
  )

  if (ARG_PUBLIC)
    set_target_properties(${tgtname}
      PROPERTIES
        Fortran_MODULE_DIRECTORY "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}"
    )
  else ()
    # Modules of different targets must not overwrite each other.
    # Ideally, we would use $<TARGET_PROPERTY:${tgtname},BINARY_DIR> but
    # Fortran_MODULE_DIRECTORY does not support generator expressions.
    # Not defining Fortran_MODULE_DIRECTORY at all would but it into
    # ${CMAKE_CURRENT_BINARY_DIR} where modules with the same name but compiled
    # by different targets would clash.
    set_target_properties(${tgtname}
      PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${tgtname}.dir/${CMAKE_CFG_INTDIR}"
    )
  endif ()
endfunction ()

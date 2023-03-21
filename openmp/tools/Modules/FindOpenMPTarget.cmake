##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Find OpenMP Target offloading Support for various compilers.
#
##===----------------------------------------------------------------------===##

#[========================================================================[.rst:
FindOpenMPTarget
----------------

Finds OpenMP Target Offloading Support.

This module can be used to detect OpenMP target offloading support in a
compiler. If the compiler support OpenMP Offloading to a specified target, the
flags required to compile offloading code to that target are output for each
target.

This module will automatically include OpenMP support if it was not loaded
already. It does not need to be included separately to get full OpenMP support.

Variables
^^^^^^^^^

The module exposes the components ``NVPTX`` and ``AMDGPU``.  Each of these
controls the various offloading targets to search OpenMP target offloasing
support for.

Depending on the enabled components the following variables will be set:

``OpenMPTarget_FOUND``
  Variable indicating that OpenMP target offloading flags for all requested
  targets have been found.

This module will set the following variables per language in your
project, where ``<device>`` is one of NVPTX or AMDGPU

``OpenMPTarget_<device>_FOUND``
  Variable indicating if OpenMP support for the ``<device>`` was detected.
``OpenMPTarget_<device>_FLAGS``
  OpenMP compiler flags for offloading to ``<device>``, separated by spaces.

For linking with OpenMP code written in ``<device>``, the following
variables are provided:

``OpenMPTarget_<device>_LIBRARIES``
  A list of libraries needed to link with OpenMP code written in ``<lang>``.

Additionally, the module provides :prop_tgt:`IMPORTED` targets:

``OpenMPTarget::OpenMPTarget_<device>``
  Target for using OpenMP offloading to ``<device>``.

If the specific architecture of the target is needed, it can be manually
specified by setting a variable to the desired architecture. Variables can also
be used to override the standard flag searching for a given compiler.

``OpenMPTarget_<device>_ARCH``
  Sets the architecture of ``<device>`` to compile for. Such as `sm_70` for NVPTX
  or `gfx908` for AMDGPU. 

``OpenMPTarget_<device>_DEVICE``
  Sets the name of the device to offload to.

``OpenMPTarget_<device>_FLAGS``
  Sets the compiler flags for offloading to ``<device>``.

#]========================================================================]

# TODO: Support Fortran
# TODO: Support multiple offloading targets by setting the "OpenMPTarget" target
#       to include flags for all components loaded
# TODO: Configure target architecture without a variable (component NVPTX_SM_70)
# TODO: Test more compilers

cmake_policy(PUSH)
cmake_policy(VERSION 3.13.4)

find_package(OpenMP ${OpenMPTarget_FIND_VERSION} REQUIRED)

# Find the offloading flags for each compiler.
function(_OPENMP_TARGET_DEVICE_FLAG_CANDIDATES LANG DEVICE)
  if(NOT OpenMPTarget_${LANG}_FLAGS)
    unset(OpenMPTarget_FLAG_CANDIDATES)

    set(OMPTarget_FLAGS_Clang "-fopenmp-targets=${DEVICE}")
    set(OMPTarget_FLAGS_GNU "-foffload=${DEVICE}=\"-lm -latomic\"")
    set(OMPTarget_FLAGS_XL "-qoffload")
    set(OMPTarget_FLAGS_PGI "-mp=${DEVICE}")
    set(OMPTarget_FLAGS_NVHPC "-mp=${DEVICE}")

    if(DEFINED OMPTarget_FLAGS_${CMAKE_${LANG}_COMPILER_ID})
      set(OpenMPTarget_FLAG_CANDIDATES "${OMPTarget_FLAGS_${CMAKE_${LANG}_COMPILER_ID}}")
    endif()

    set(OpenMPTarget_${LANG}_FLAG_CANDIDATES "${OpenMPTarget_FLAG_CANDIDATES}" PARENT_SCOPE)
  else()
    set(OpenMPTarget_${LANG}_FLAG_CANDIDATES "${OpenMPTarget_${LANG}_FLAGS}" PARENT_SCOPE)
  endif()
endfunction()

# Get the coded name of the device for each compiler.
function(_OPENMP_TARGET_DEVICE_CANDIDATES LANG DEVICE)
  if (NOT OpenMPTarget_${DEVICE}_DEVICE)
    unset(OpenMPTarget_DEVICE_CANDIDATES)

    # Check each supported device.
    if("${DEVICE}" STREQUAL "NVPTX")
      if ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
        set(OMPTarget_DEVICE_Clang "nvptx32-nvidia-cuda")
      else()
        set(OMPTarget_DEVICE_Clang "nvptx64-nvidia-cuda")
      endif()
      set(OMPTarget_DEVICE_GNU "nvptx-none")
      set(OMPTarget_DEVICE_XL "")
      set(OMPTarget_DEVICE_PGI "gpu")
      set(OMPTarget_DEVICE_NVHPC "gpu")

      if(DEFINED OMPTarget_DEVICE_${CMAKE_${LANG}_COMPILER_ID})
        set(OpenMPTarget_DEVICE_CANDIDATES "${OMPTarget_DEVICE_${CMAKE_${LANG}_COMPILER_ID}}")
      endif()
    elseif("${DEVICE}" STREQUAL "AMDGPU")
      set(OMPTarget_DEVICE_Clang "amdgcn-amd-amdhsa")
      set(OMPTarget_DEVICE_GNU "hsa")

      if(DEFINED OMPTarget_DEVICE_${CMAKE_${LANG}_COMPILER_ID})
        set(OpenMPTarget_DEVICE_CANDIDATES "${OMPTarget_DEVICE_${CMAKE_${LANG}_COMPILER_ID}}")
      endif()
    endif()
    set(OpenMPTarget_${LANG}_DEVICE_CANDIDATES "${OpenMPTarget_DEVICE_CANDIDATES}" PARENT_SCOPE)
  else()
    set(OpenMPTarget_${LANG}_DEVICE_CANDIDATES "${OpenMPTarget_${LANG}_DEVICE}" PARENT_SCOPE)
  endif()
endfunction()

# Get flags for setting the device's architecture for each compiler.
function(_OPENMP_TARGET_DEVICE_ARCH_CANDIDATES LANG DEVICE DEVICE_FLAG)
  if(OpenMPTarget_${DEVICE}_ARCH)
    # Only Clang supports selecting the architecture for now.
    set(OMPTarget_ARCH_Clang "-Xopenmp-target=${DEVICE_FLAG} -march=${OpenMPTarget_${DEVICE}_ARCH}")

    if(DEFINED OMPTarget_ARCH_${CMAKE_${LANG}_COMPILER_ID})
      set(OpenMPTarget_DEVICE_ARCH_CANDIDATES "${OMPTarget_ARCH_${CMAKE_${LANG}_COMPILER_ID}}")
    endif()
    set(OpenMPTarget_${LANG}_DEVICE_ARCH_CANDIDATES "${OpenMPTarget_DEVICE_ARCH_CANDIDATES}" PARENT_SCOPE)
  else()
    set(OpenMPTarget_${LANG}_DEVICE_ARCH_CANDIDATES "" PARENT_SCOPE)
  endif()
endfunction()

set(OpenMPTarget_C_CXX_TEST_SOURCE
"#include <omp.h>
int main(void) {
  int isHost;
#pragma omp target map(from: isHost)
  { isHost = omp_is_initial_device(); }
  return isHost;
}")

function(_OPENMP_TARGET_WRITE_SOURCE_FILE LANG SRC_FILE_CONTENT_VAR SRC_FILE_NAME SRC_FILE_FULLPATH)
  set(WORK_DIR ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FindOpenMPTarget)
  if("${LANG}" STREQUAL "C")
    set(SRC_FILE "${WORK_DIR}/${SRC_FILE_NAME}.c")
    file(WRITE "${SRC_FILE}" "${OpenMPTarget_C_CXX_${SRC_FILE_CONTENT_VAR}}")
  elseif("${LANG}" STREQUAL "CXX")
    set(SRC_FILE "${WORK_DIR}/${SRC_FILE_NAME}.cpp")
    file(WRITE "${SRC_FILE}" "${OpenMPTarget_C_CXX_${SRC_FILE_CONTENT_VAR}}")
  endif()
  set(${SRC_FILE_FULLPATH} "${SRC_FILE}" PARENT_SCOPE)
endfunction()

# Get the candidate flags and try to compile a test application. If it compiles
# and all the flags are found, we assume the compiler supports offloading.
function(_OPENMP_TARGET_DEVICE_GET_FLAGS LANG DEVICE OPENMP_FLAG_VAR OPENMP_LIB_VAR OPENMP_DEVICE_VAR OPENMP_ARCH_VAR)
  _OPENMP_TARGET_DEVICE_CANDIDATES(${LANG} ${DEVICE})
  _OPENMP_TARGET_DEVICE_FLAG_CANDIDATES(${LANG} "${OpenMPTarget_${LANG}_DEVICE_CANDIDATES}")
  _OPENMP_TARGET_DEVICE_ARCH_CANDIDATES(${LANG} ${DEVICE} "${OpenMPTarget_${LANG}_DEVICE_CANDIDATES}")
  _OPENMP_TARGET_WRITE_SOURCE_FILE("${LANG}" "TEST_SOURCE" OpenMPTargetTryFlag _OPENMP_TEST_SRC)

  # Try to compile a test application with the found flags.
  try_compile(OpenMPTarget_COMPILE_RESULT ${CMAKE_BINARY_DIR} ${_OPENMP_TEST_SRC}
    CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${OpenMP_${LANG}_FLAGS} ${OpenMPTarget_${LANG}_FLAG_CANDIDATES} ${OpenMPTarget_${LANG}_DEVICE_ARCH_CANDIDATES}"
    "-DINCLUDE_DIRECTORIES:STRING=${OpenMP_${LANG}_INCLUDE_DIR}"
    LINK_LIBRARIES ${CMAKE_${LANG}_VERBOSE_FLAG}
    OUTPUT_VARIABLE OpenMP_TRY_COMPILE_OUTPUT
  )

  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
    "Detecting OpenMP ${CMAKE_${LANG}_COMPILER_ID} ${DEVICE} target support with the following Flags:
    ${OpenMP_${LANG}_FLAGS} ${OpenMPTarget_${LANG}_FLAG_CANDIDATES} ${OpenMPTarget_${LANG}_DEVICE_ARCH_CANDIDATES}
    With the following output:\n ${OpenMP_TRY_COMPILE_OUTPUT}\n")

  # If compilation was successful and the device was found set the return variables.
  if (OpenMPTarget_COMPILE_RESULT AND DEFINED OpenMPTarget_${LANG}_DEVICE_CANDIDATES)
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Compilation successful, adding flags for ${DEVICE}.\n\n")

    # Clang has a seperate library for target offloading.
    if(CMAKE_${LANG}_COMPILER_ID STREQUAL "Clang")
      find_library(OpenMPTarget_libomptarget_LIBRARY
        NAMES omptarget
        HINTS ${CMAKE_${LANG}_IMPLICIT_LINK_DIRECTORIES}
      )
      mark_as_advanced(OpenMPTarget_libomptarget_LIBRARY)
      set("${OPENMP_LIB_VAR}" "${OpenMPTarget_libomptarget_LIBRARY}" PARENT_SCOPE)
    else()
      unset("${OPENMP_LIB_VAR}" PARENT_SCOPE)
    endif()
    set("${OPENMP_DEVICE_VAR}" "${OpenMPTarget_${LANG}_DEVICE_CANDIDATES}" PARENT_SCOPE)
    set("${OPENMP_FLAG_VAR}" "${OpenMPTarget_${LANG}_FLAG_CANDIDATES}" PARENT_SCOPE)
    set("${OPENMP_ARCH_VAR}" "${OpenMPTarget_${LANG}_DEVICE_ARCH_CANDIDATES}" PARENT_SCOPE)
  else()
    unset("${OPENMP_DEVICE_VAR}" PARENT_SCOPE)
    unset("${OPENMP_FLAG_VAR}" PARENT_SCOPE)
    unset("${OPENMP_ARCH_VAR}" PARENT_SCOPE)
  endif()
endfunction()

# Load the compiler support for each device.
foreach(LANG IN ITEMS C CXX)
  # Cache the version in case CMake doesn't load the OpenMP package this time
  set(OpenMP_${LANG}_VERSION ${OpenMP_${LANG}_VERSION}
    CACHE STRING "OpenMP Version" FORCE)
  mark_as_advanced(OpenMP_${LANG}_VERSION)
  foreach(DEVICE IN ITEMS NVPTX AMDGPU)
    if(CMAKE_${LANG}_COMPILER_LOADED)
      if(NOT DEFINED OpenMPTarget_${LANG}_FLAGS OR NOT DEFINED OpenMPTarget_${LANG}_DEVICE)
        _OPENMP_TARGET_DEVICE_GET_FLAGS(${LANG} ${DEVICE}
          OpenMPTarget_${DEVICE}_FLAGS_WORK
          OpenMPTarget_${DEVICE}_LIBS_WORK
          OpenMPTarget_${DEVICE}_DEVICE_WORK
          OpenMPTarget_${DEVICE}_ARCHS_WORK)

        separate_arguments(_OpenMPTarget_${DEVICE}_FLAGS NATIVE_COMMAND "${OpenMPTarget_${DEVICE}_FLAGS_WORK}")
        separate_arguments(_OpenMPTarget_${DEVICE}_ARCHS NATIVE_COMMAND "${OpenMPTarget_${DEVICE}_ARCHS_WORK}")
        set(OpenMPTarget_${DEVICE}_FLAGS ${_OpenMPTarget_${DEVICE}_FLAGS}
            CACHE STRING "${DEVICE} target compile flags for OpenMP target offloading" FORCE)
        set(OpenMPTarget_${DEVICE}_ARCH ${_OpenMPTarget_${DEVICE}_ARCHS}
            CACHE STRING "${DEVICE} target architecture flags for OpenMP target offloading" FORCE)
        set(OpenMPTarget_${DEVICE}_LIBRARIES ${OpenMPTarget_${DEVICE}_LIBS_WORK}
            CACHE STRING "${DEVICE} target libraries for OpenMP target offloading" FORCE)
        mark_as_advanced(OpenMPTarget_${DEVICE}_FLAGS OpenMPTarget_${DEVICE}_ARCH OpenMPTarget_${DEVICE}_LIBRARIES)
      endif()
    endif()
  endforeach()
endforeach()

if(OpenMPTarget_FIND_COMPONENTS)
  set(OpenMPTarget_FINDLIST ${OpenMPTarget_FIND_COMPONENTS})
else()
  set(OpenMPTarget_FINDLIST NVPTX)
endif()

unset(_OpenMPTarget_MIN_VERSION)

# Attempt to find each requested device.
foreach(LANG IN ITEMS C CXX)
  foreach(DEVICE IN LISTS OpenMPTarget_FINDLIST)
    if(CMAKE_${LANG}_COMPILER_LOADED)
      set(OpenMPTarget_${DEVICE}_VERSION "${OpenMP_${LANG}_VERSION}")
      set(OpenMPTarget_${DEVICE}_VERSION_MAJOR "${OpenMP_${LANG}_VERSION}_MAJOR")
      set(OpenMPTarget_${DEVICE}_VERSION_MINOR "${OpenMP_${LANG}_VERSION}_MINOR")
      set(OpenMPTarget_${DEVICE}_FIND_QUIETLY ${OpenMPTarget_FIND_QUIETLY})
      set(OpenMPTarget_${DEVICE}_FIND_REQUIRED ${OpenMPTarget_FIND_REQUIRED})
      set(OpenMPTarget_${DEVICE}_FIND_VERSION ${OpenMPTarget_FIND_VERSION})
      set(OpenMPTarget_${DEVICE}_FIND_VERSION_EXACT ${OpenMPTarget_FIND_VERSION_EXACT})

      # OpenMP target offloading is only supported in OpenMP 4.0 an newer.
      if(OpenMPTarget_${DEVICE}_VERSION AND ("${OpenMPTarget_${DEVICE}_VERSION}" VERSION_LESS "4.0"))
        message(SEND_ERROR "FindOpenMPTarget requires at least OpenMP 4.0")
      endif()

      set(FPHSA_NAME_MISMATCHED TRUE)
      find_package_handle_standard_args(OpenMPTarget_${DEVICE}
        REQUIRED_VARS OpenMPTarget_${DEVICE}_FLAGS
        VERSION_VAR OpenMPTarget_${DEVICE}_VERSION)

      if(OpenMPTarget_${DEVICE}_FOUND)
        if(DEFINED OpenMPTarget_${DEVICE}_VERSION)
          if(NOT _OpenMPTarget_MIN_VERSION OR _OpenMPTarget_MIN_VERSION VERSION_GREATER OpenMPTarget_${LANG}_VERSION)
            set(_OpenMPTarget_MIN_VERSION OpenMPTarget_${DEVICE}_VERSION)
          endif()
        endif()
        # Create a new target.
        if(NOT TARGET OpenMPTarget::OpenMPTarget_${DEVICE})
          add_library(OpenMPTarget::OpenMPTarget_${DEVICE} INTERFACE IMPORTED)
        endif()
        # Get compiler flags for offloading to the device and architecture and
        # set the target features. Include the normal OpenMP flags as well.
        set_property(TARGET OpenMPTarget::OpenMPTarget_${DEVICE} PROPERTY
          INTERFACE_COMPILE_OPTIONS 
          "$<$<COMPILE_LANGUAGE:${LANG}>:${OpenMPTarget_${DEVICE}_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:${LANG}>:${OpenMPTarget_${DEVICE}_ARCH}>"
          "$<$<COMPILE_LANGUAGE:${LANG}>:${OpenMP_${LANG}_FLAGS}>")
        set_property(TARGET OpenMPTarget::OpenMPTarget_${DEVICE} PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${OpenMP_${LANG}_INCLUDE_DIRS}>")
        set_property(TARGET OpenMPTarget::OpenMPTarget_${DEVICE} PROPERTY
          INTERFACE_LINK_LIBRARIES 
          "${OpenMPTarget_${DEVICE}_LIBRARIES}"
          "${OpenMP_${LANG}_LIBRARIES}")
        # The offloading flags must also be passed during the linking phase so
        # the compiler can pass the binary to the correct toolchain.
        set_property(TARGET OpenMPTarget::OpenMPTarget_${DEVICE} PROPERTY
          INTERFACE_LINK_OPTIONS 
          "$<$<COMPILE_LANGUAGE:${LANG}>:${OpenMPTarget_${DEVICE}_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:${LANG}>:${OpenMPTarget_${DEVICE}_ARCH}>"
          "$<$<COMPILE_LANGUAGE:${LANG}>:${OpenMP_${LANG}_FLAGS}>")
        # Combine all the flags if not using the target for convenience.
        set(OpenMPTarget_${DEVICE}_FLAGS ${OpenMP_${LANG}_FLAGS}
          ${OpenMPTarget_${DEVICE}_FLAGS} 
          ${OpenMPTarget_${DEVICE}_ARCH}
          CACHE STRING "${DEVICE} target compile flags for OpenMP target offloading" FORCE)
      endif()
    endif()
  endforeach()
endforeach()

unset(_OpenMPTarget_REQ_VARS)
foreach(DEVICE IN LISTS OpenMPTarget_FINDLIST)
  list(APPEND _OpenMPTarget_REQ_VARS "OpenMPTarget_${DEVICE}_FOUND")
endforeach()

find_package_handle_standard_args(OpenMPTarget
    REQUIRED_VARS ${_OpenMPTarget_REQ_VARS}
    VERSION_VAR ${_OpenMPTarget_MIN_VERSION}
    HANDLE_COMPONENTS)

if(NOT (CMAKE_C_COMPILER_LOADED OR CMAKE_CXX_COMPILER_LOADED) OR CMAKE_Fortran_COMPILER_LOADED)
  message(SEND_ERROR "FindOpenMPTarget requires the C or CXX languages to be enabled")
endif()

unset(OpenMPTarget_C_CXX_TEST_SOURCE)
cmake_policy(POP)

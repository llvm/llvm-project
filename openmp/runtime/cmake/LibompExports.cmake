#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

# LibompExports.cmake
#   Copy library and header files into the exports/ subdirectory after library build

# Create the suffix for the export directory
# - Only add to suffix when not a default value
# - Example suffix: .deb.s1
#   final export directory: exports/lin_32e.deb.s1/lib
# - These suffixes imply the build is a Debug, Stats-Gathering version of the library
set(libomp_suffix)
libomp_append(libomp_suffix .deb DEBUG_BUILD)
libomp_append(libomp_suffix .dia RELWITHDEBINFO_BUILD)
libomp_append(libomp_suffix .min MINSIZEREL_BUILD)
libomp_append(libomp_suffix .s1 LIBOMP_STATS)
libomp_append(libomp_suffix .ompt LIBOMP_OMPT_SUPPORT)
if(${LIBOMP_OMPT_SUPPORT})
  libomp_append(libomp_suffix .optional LIBOMP_OMPT_OPTIONAL)
endif()
string(REPLACE ";" "" libomp_suffix "${libomp_suffix}")

# Set exports locations
if(WIN32)
  set(LIBOMP_SHORT_OS win)
elseif(APPLE)
  set(LIBOMP_SHORT_OS mac)
else()
  set(LIBOMP_SHORT_OS lin)
endif()
if(${MIC})
  set(libomp_platform "${LIBOMP_SHORT_OS}_${LIBOMP_MIC_ARCH}") # e.g., lin_knf, lin_knc
else()
  if(${IA32})
    set(libomp_platform "${LIBOMP_SHORT_OS}_32")
  elseif(${INTEL64})
    set(libomp_platform "${LIBOMP_SHORT_OS}_32e")
  else()
    set(libomp_platform "${LIBOMP_SHORT_OS}_${LIBOMP_ARCH}") # e.g., lin_arm, lin_ppc64
  endif()
endif()
set(LIBOMP_EXPORTS_DIR "${LIBOMP_BASE_DIR}/exports")
set(LIBOMP_EXPORTS_PLATFORM_DIR "${LIBOMP_EXPORTS_DIR}/${libomp_platform}${libomp_suffix}")
set(LIBOMP_EXPORTS_CMN_DIR "${LIBOMP_EXPORTS_DIR}/common${libomp_suffix}/include")
set(LIBOMP_EXPORTS_INC_DIR "${LIBOMP_EXPORTS_PLATFORM_DIR}/include")
set(LIBOMP_EXPORTS_MOD_DIR "${LIBOMP_EXPORTS_PLATFORM_DIR}/include_compat")
set(LIBOMP_EXPORTS_LIB_DIR "${LIBOMP_EXPORTS_DIR}/${libomp_platform}${libomp_suffix}/lib")

# Put headers in exports/ directory post build
add_custom_command(TARGET omp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBOMP_EXPORTS_CMN_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy omp.h ${LIBOMP_EXPORTS_CMN_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ompx.h ${LIBOMP_EXPORTS_CMN_DIR}
)
if(${LIBOMP_OMPT_SUPPORT})
  add_custom_command(TARGET omp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy omp-tools.h ${LIBOMP_EXPORTS_CMN_DIR}
  )
endif()
if(${LIBOMP_FORTRAN_MODULES})
  add_custom_command(TARGET libomp-mod POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBOMP_EXPORTS_MOD_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy omp_lib.mod ${LIBOMP_EXPORTS_MOD_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy omp_lib_kinds.mod ${LIBOMP_EXPORTS_MOD_DIR}
  )
  add_custom_command(TARGET omp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy omp_lib.h ${LIBOMP_EXPORTS_CMN_DIR}
  )
endif()

# Copy OpenMP library into exports/ directory post build
if(WIN32)
  get_target_property(LIBOMP_OUTPUT_DIRECTORY omp RUNTIME_OUTPUT_DIRECTORY)
else()
  get_target_property(LIBOMP_OUTPUT_DIRECTORY omp LIBRARY_OUTPUT_DIRECTORY)
endif()
if(NOT LIBOMP_OUTPUT_DIRECTORY)
  set(LIBOMP_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
add_custom_command(TARGET omp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBOMP_EXPORTS_LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:omp> ${LIBOMP_EXPORTS_LIB_DIR}
)

# Copy Windows import library into exports/ directory post build
if(WIN32)
  get_target_property(LIBOMPIMP_OUTPUT_DIRECTORY ${LIBOMP_IMP_LIB_TARGET} ARCHIVE_OUTPUT_DIRECTORY)
  if(NOT LIBOMPIMP_OUTPUT_DIRECTORY)
    set(LIBOMPIMP_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
  add_custom_command(TARGET ${LIBOMP_IMP_LIB_TARGET} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBOMP_EXPORTS_LIB_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${LIBOMPIMP_OUTPUT_DIRECTORY}/${LIBOMP_IMP_LIB_FILE} ${LIBOMP_EXPORTS_LIB_DIR}
  )
endif()

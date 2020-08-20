##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# llvm-project/openmp/libomptarget/hostrpc/CMakeLists.txt
# Written by Greg Rodgers (Gregory.Rodgers@amd.com)
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
add_subdirectory(services)
# find the llvm compiler
find_package(LLVM QUIET CONFIG
  PATHS
  $ENV{AOMP}
  ${CMAKE_CXX_COMPILER_DIR}
  NO_DEFAULT_PATH
)

if (LLVM_DIR)
  message("   -- Building hostrpc with LLVM ${LLVM_PACKAGE_VERSION} found at ${LLVM_INSTALL_PREFIX}")
else()
  message("   ERROR: NO LLVM FOUND! Not building hostrpc .")
  return()
endif()

if(${AOMP_STANDALONE_BUILD} STREQUAL "0")
  message("------------- NON STANDALONE BUILD ---------------")
  set(ROCDL_INC_OCKL ${DEVICELIBS_ROOT}/ockl/inc)
  set(ROCDL_INC_OCML ${DEVICELIBS_ROOT}/ocml/inc)
  set(ROCDL_INC_IRIF ${DEVICELIBS_ROOT}/irif/inc)
  set(HIPINC ${ROCM_DIR}/hip/include)

else()
  # Assome rocm-device-libs repository is next to aomp-extras repository
  set(ROCDL ${CMAKE_CURRENT_SOURCE_DIR}/../../../../rocm-device-libs)
  set(ROCDL_INC_OCKL ${ROCDL}/ockl/inc)
  set(ROCDL_INC_OCML ${ROCDL}/ocml/inc)
  set(ROCDL_INC_IRIF ${ROCDL}/irif/inc)
  set(HIPINC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../hip-on-vdi/include)
endif()

# add_custom_target(hostrpc ALL)

#optimization level
set(optimization_level 2)

set(sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hostrpc.cpp
)

# for both in-tree and out-of-tree build
if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
else()
  set(OUTPUTDIR ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endif()

macro(collect_sources dir)
  set(openmp_sources)
  foreach(file ${ARGN})
    file(RELATIVE_PATH rfile ${dir} ${file})
    get_filename_component(rdir ${rfile} DIRECTORY)
    get_filename_component(fname ${rfile} NAME_WE)
    get_filename_component(fext ${rfile} EXT)
    #file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${rdir})
    if (fext STREQUAL ".cpp")
      set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${rdir}/${fname}.cpp)
      list(APPEND openmp_sources ${cfile})
    endif()
    if (fext STREQUAL ".c")
      set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${rdir}/${fname}.c)
      list(APPEND openmp_sources ${cfile})
    endif()
  endforeach()
endmacro()

macro(add_openmp_library libname archname dir)
  if (archname  STREQUAL "amdgcn")
    set(mcpus $ENV{GFXLIST})
    if(mcpus)
      separate_arguments(mcpus)
      list(GET mcpus 0 GPU)
    else()
      set(GPU "gfx803")
    endif()
    set(triple "amdgcn-amd-amdhsa")
  else()
    set(GPU "sm_70")
    set(triple "nvptx-nvidia-cuda")
  endif()

  set(openmp_cmd ${LLVM_INSTALL_PREFIX}/bin/clang++)
  set(openmp_c_cmd ${LLVM_INSTALL_PREFIX}/bin/clang)
  set(openmp_args -O${optimization_level}
    -fopenmp 
    -fopenmp-targets=${triple}
    -Xopenmp-target=${triple}
    -march=${GPU}
    -c 
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${ROCM_DIR}/include
    -I${CMAKE_CURRENT_BINARY_DIR}/../../runtime/src
    -I${ROCDL_INC_OCKL}
    -I${ROCDL_INC_OCML}
    -I${ROCDL_INC_IRIF})

  if(NOT ${AOMP_STANDALONE_BUILD})
    set(openmp_args "${openmp_args};--hip-device-lib-path=${ROCM_DIR}/amdgcn/bitcode")
  endif(NOT ${AOMP_STANDALONE_BUILD})
  set(obj_files)
  set(bc_files)
  file(GLOB h_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
  foreach(file ${ARGN})
    file(RELATIVE_PATH rfile ${dir} ${file})
    get_filename_component(rdir ${rfile} DIRECTORY)
    get_filename_component(fname ${rfile} NAME_WE)
    get_filename_component(fext ${rfile} EXT)

    set(obj_filename "${fname}.o")
    set(bc_filename "${fname}-${archname}.bc")

    if (fext STREQUAL ".c")
      add_custom_command(
        OUTPUT ${obj_filename}
        COMMAND ${openmp_c_cmd} ${openmp_args} ${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.c -o ${obj_filename}
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.c" ${h_files} )
    else()
      add_custom_command(
        OUTPUT ${obj_filename}
        COMMAND ${openmp_cmd} ${openmp_args} -std=c++11 ${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.cpp -o ${obj_filename}
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.cpp" ${h_files} )
    endif()

    # Extract the bitcode file to make aarch specific bitcode archive
    add_custom_command(
      OUTPUT ${bc_filename}
      COMMAND ${LLVM_INSTALL_PREFIX}/bin/clang-offload-bundler -type=o -targets=openmp-${triple}-${GPU} -inputs=${obj_filename} -outputs=${bc_filename} -unbundle
      DEPENDS ${obj_filename} )

    list(APPEND obj_files ${obj_filename})
    list(APPEND bc_files ${bc_filename})
  endforeach()

  set(host_ar_filename "lib${libname}.a")
  set(bc_ar_filename "libbc-${libname}-${archname}.a")
  string(REPLACE ";" " " _objfiles_with_spaces ${obj_files})
  string(REPLACE ";" " " _bcfiles_with_spaces ${bc_files})
  add_custom_command(
      OUTPUT ${host_ar_filename}
      COMMAND ${LLVM_INSTALL_PREFIX}/bin/llvm-ar rcs ${host_ar_filename} ${_objfiles_with_spaces}
      DEPENDS ${obj_files} )
  add_custom_command(
      OUTPUT ${bc_ar_filename}
      COMMAND ${LLVM_INSTALL_PREFIX}/bin/llvm-ar rcs ${bc_ar_filename} ${_bcfiles_with_spaces}
      DEPENDS ${bc_files} )

  add_custom_target(lib${name}-host-static-lib ALL DEPENDS ${host_ar_filename})
  add_custom_target(lib${name}-device-static-lib ALL DEPENDS ${bc_ar_filename})
endmacro()

collect_sources(${CMAKE_CURRENT_SOURCE_DIR}/src ${sources})

set(libname "hostrpc")
set(archname "amdgcn")
add_openmp_library(${libname} ${archname} ${CMAKE_CURRENT_SOURCE_DIR} ${openmp_sources})

install(FILES ${OUTPUTDIR}/hostrpc/lib${libname}.a DESTINATION "lib")
install(FILES ${OUTPUTDIR}/hostrpc/lib${libname}.a DESTINATION "lib-debug")
# GPU architecture-specific archives prevent need for unbundle archive
install(FILES ${OUTPUTDIR}/hostrpc/libbc-${libname}-${archname}.a DESTINATION "lib/libdevice")
install(FILES ${OUTPUTDIR}/hostrpc/libbc-${libname}-${archname}.a DESTINATION "lib-debug/libdevice")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/hostrpc.h" DESTINATION include)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  message("------ DONE hostrpc cmake -------")
endif()

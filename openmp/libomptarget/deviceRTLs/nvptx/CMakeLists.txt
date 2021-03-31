##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build the NVPTX (CUDA) Device RTL if the CUDA tools are available
#
##===----------------------------------------------------------------------===##

# By default we will not build NVPTX deviceRTL on a CUDA free system
set(LIBOMPTARGET_BUILD_NVPTX_BCLIB FALSE CACHE BOOL
  "Whether build NVPTX deviceRTL on CUDA free system.")

if (NOT (LIBOMPTARGET_DEP_CUDA_FOUND OR LIBOMPTARGET_BUILD_NVPTX_BCLIB))
  libomptarget_say("Not building NVPTX deviceRTL by default on CUDA free system.")
  return()
endif()

# Check if we can create an LLVM bitcode implementation of the runtime library
# that could be inlined in the user application. For that we need to find
# a Clang compiler capable of compiling our CUDA files to LLVM bitcode and
# an LLVM linker.
set(LIBOMPTARGET_NVPTX_CUDA_COMPILER "" CACHE STRING
  "Location of a CUDA compiler capable of emitting LLVM bitcode.")
set(LIBOMPTARGET_NVPTX_BC_LINKER "" CACHE STRING
  "Location of a linker capable of linking LLVM bitcode objects.")

if (NOT LIBOMPTARGET_NVPTX_CUDA_COMPILER STREQUAL "")
  set(cuda_compiler ${LIBOMPTARGET_NVPTX_CUDA_COMPILER})
elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
  set(cuda_compiler ${CMAKE_C_COMPILER})
else()
  libomptarget_say("Not building NVPTX deviceRTL: clang not found")
  return()
endif()

# Get compiler directory to try to locate a suitable linker.
get_filename_component(compiler_dir ${cuda_compiler} DIRECTORY)
set(llvm_link "${compiler_dir}/llvm-link")

if (NOT LIBOMPTARGET_NVPTX_BC_LINKER STREQUAL "")
  set(bc_linker ${LIBOMPTARGET_NVPTX_BC_LINKER})
elseif (EXISTS ${llvm_link})
  set(bc_linker ${llvm_link})
else()
  libomptarget_say("Not building NVPTX deviceRTL: llvm-link not found")
  return()
endif()

# TODO: This part needs to be refined when libomptarget is going to support
# Windows!
# TODO: This part can also be removed if we can change the clang driver to make
# it support device only compilation.
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(aux_triple x86_64-unknown-linux-gnu)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "ppc64le")
  set(aux_triple powerpc64le-unknown-linux-gnu)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(aux_triple aarch64-unknown-linux-gnu)
else()
  libomptarget_say("Not building CUDA offloading device RTL: unknown host arch: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
  return()
endif()

get_filename_component(devicertl_base_directory
  ${CMAKE_CURRENT_SOURCE_DIR}
  DIRECTORY)
set(devicertl_common_directory
  ${devicertl_base_directory}/common)
set(devicertl_nvptx_directory
  ${devicertl_base_directory}/nvptx)

set(all_capabilities 35 37 50 52 53 60 61 62 70 72 75 80)

set(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES ${all_capabilities} CACHE STRING
  "List of CUDA Compute Capabilities to be used to compile the NVPTX device RTL.")
string(TOLOWER ${LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES} LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES)

if (LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES STREQUAL "all")
  set(nvptx_sm_list ${all_capabilities})
elseif(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES STREQUAL "auto")
  if (NOT LIBOMPTARGET_DEP_CUDA_FOUND)
    libomptarget_error_say("[NVPTX] Cannot auto detect compute capability as CUDA not found.")
  endif()
  set(nvptx_sm_list ${LIBOMPTARGET_DEP_CUDA_ARCH})
else()
  string(REPLACE "," ";" nvptx_sm_list "${LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES}")
endif()

# If user set LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES to empty, we disable the
# build.
if (NOT nvptx_sm_list)
  libomptarget_say("Not building CUDA offloading device RTL: empty compute capability list")
  return()
endif()

# Check all SM values
foreach(sm ${nvptx_sm_list})
  if (NOT ${sm} IN_LIST all_capabilities)
    libomptarget_warning_say("[NVPTX] Compute capability ${sm} is not supported. Make sure clang can work with it.")
  endif()
endforeach()

# Override default MAX_SM in src/target_impl.h if requested
if (DEFINED LIBOMPTARGET_NVPTX_MAX_SM)
  set(MAX_SM_DEFINITION "-DMAX_SM=${LIBOMPTARGET_NVPTX_MAX_SM}")
endif()

# Activate RTL message dumps if requested by the user.
set(LIBOMPTARGET_NVPTX_DEBUG FALSE CACHE BOOL
  "Activate NVPTX device RTL debug messages.")

libomptarget_say("Building CUDA LLVM bitcode offloading device RTL.")

set(cuda_src_files
  ${devicertl_common_directory}/src/cancel.cu
  ${devicertl_common_directory}/src/critical.cu
  ${devicertl_common_directory}/src/data_sharing.cu
  ${devicertl_common_directory}/src/libcall.cu
  ${devicertl_common_directory}/src/loop.cu
  ${devicertl_common_directory}/src/omp_data.cu
  ${devicertl_common_directory}/src/omptarget.cu
  ${devicertl_common_directory}/src/parallel.cu
  ${devicertl_common_directory}/src/reduction.cu
  ${devicertl_common_directory}/src/support.cu
  ${devicertl_common_directory}/src/sync.cu
  ${devicertl_common_directory}/src/task.cu
  ${devicertl_common_directory}/src/shuffle.cpp
  src/target_impl.cu
)

# Set flags for LLVM Bitcode compilation.
set(bc_flags -S -x c++ -O1 -std=c++14
             -target nvptx64
             -Xclang -emit-llvm-bc
             -Xclang -aux-triple -Xclang ${aux_triple}
             -fopenmp -fopenmp-cuda-mode -Xclang -fopenmp-is-device
             -Xclang -target-feature -Xclang +ptx61
             -D__CUDACC__
             -I${devicertl_base_directory}
             -I${devicertl_common_directory}/include
             -I${devicertl_nvptx_directory}/src)

if(${LIBOMPTARGET_NVPTX_DEBUG})
  list(APPEND bc_flags -DOMPTARGET_NVPTX_DEBUG=-1)
else()
  list(APPEND bc_flags -DOMPTARGET_NVPTX_DEBUG=0)
endif()

# Create target to build all Bitcode libraries.
add_custom_target(omptarget-nvptx-bc)

# Generate a Bitcode library for all the compute capabilities the user requested
foreach(sm ${nvptx_sm_list})
  set(cuda_flags -Xclang -target-cpu -Xclang sm_${sm} "-D__CUDA_ARCH__=${sm}0")
  set(bc_files "")
  foreach(src ${cuda_src_files})
    get_filename_component(infile ${src} ABSOLUTE)
    get_filename_component(outfile ${src} NAME)
    set(outfile "${outfile}-sm_${sm}.bc")

    add_custom_command(OUTPUT ${outfile}
      COMMAND ${cuda_compiler} ${bc_flags}
        ${cuda_flags} ${MAX_SM_DEFINITION} ${infile} -o ${outfile}
      DEPENDS ${infile}
      IMPLICIT_DEPENDS CXX ${infile}
      COMMENT "Building LLVM bitcode ${outfile}"
      VERBATIM
    )
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${outfile})

    list(APPEND bc_files ${outfile})
  endforeach()

  set(bclib_name "libomptarget-nvptx-sm_${sm}.bc")

  # Link to a bitcode library.
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      COMMAND ${bc_linker}
        -o ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name} ${bc_files}
      DEPENDS ${bc_files}
      COMMENT "Linking LLVM bitcode ${bclib_name}"
  )
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${bclib_name})

  set(bclib_target_name "omptarget-nvptx-sm_${sm}-bc")

  add_custom_target(${bclib_target_name} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name})
  add_dependencies(omptarget-nvptx-bc ${bclib_target_name})

  # Copy library to destination.
  add_custom_command(TARGET ${bclib_target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
                    ${LIBOMPTARGET_LIBRARY_DIR})

  # Install bitcode library under the lib destination folder.
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name} DESTINATION "${OPENMP_INSTALL_LIBDIR}")
endforeach()

# Test will be enabled if the building machine supports CUDA
if (LIBOMPTARGET_DEP_CUDA_FOUND)
  add_subdirectory(test)
endif()

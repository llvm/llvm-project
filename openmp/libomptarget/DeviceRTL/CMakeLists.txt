##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build the Device RTL for all toolchains that are available
#
##===----------------------------------------------------------------------===##

# TODO: copied from NVPTX, need to be generalized.

# By default we will not build NVPTX deviceRTL on a CUDA free system
set(LIBOMPTARGET_BUILD_NVPTX_BCLIB FALSE CACHE BOOL
  "Whether build NVPTX deviceRTL on CUDA free system.")

if (NOT (LIBOMPTARGET_DEP_CUDA_FOUND OR LIBOMPTARGET_BUILD_NVPTX_BCLIB))
  libomptarget_say("Not building NVPTX deviceRTL by default on CUDA free system.")
  return()
endif()

if (NOT LIBOMPTARGET_LLVM_INCLUDE_DIRS)
  libomptarget_say("Not building device RTL: Missing definition for LIBOMPTARGET_LLVM_INCLUDE_DIRS")
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
elseif (LLVM_TOOL_CLANG_BUILD AND NOT CMAKE_CROSSCOMPILING)
  # Compile the deviceRTL with the clang that is built in the project.
  set(cuda_compiler "$<TARGET_FILE:clang>")
elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
  set(cuda_compiler ${CMAKE_C_COMPILER})
else()
  libomptarget_say("Not building deviceRTL: clang not found")
  return()
endif()

# Get compiler directory to try to locate a suitable linker.
get_filename_component(compiler_dir ${cuda_compiler} DIRECTORY)

set(bc_linker_candidate "${compiler_dir}/llvm-link")
if (NOT LIBOMPTARGET_NVPTX_BC_LINKER STREQUAL "")
  set(bc_linker ${LIBOMPTARGET_NVPTX_BC_LINKER})
elseif (EXISTS "${bc_linker_candidate}" AND NOT IS_DIRECTORY "${bc_linker_candidate}")
  # Try to use the linker consistent with the CUDA compiler unless explicitly
  # set to a different linker.
  set(bc_linker "${bc_linker_candidate}")
elseif (NOT OPENMP_STANDALONE_BUILD AND NOT CMAKE_CROSSCOMPILING)
  # Use the linker also built in the same project.
  set(bc_linker "$<TARGET_FILE:llvm-link>")
else()
  libomptarget_say("Not building deviceRTL: llvm-link not found")
  return()
endif()

set(opt_candidate "${compiler_dir}/opt")
if (EXISTS "${opt_candidate}" AND NOT IS_DIRECTORY "${opt_candidate}")
  # Try to use the opt consistent with the CUDA compiler.
  set(opt "${opt_candidate}")
elseif (NOT OPENMP_STANDALONE_BUILD AND NOT CMAKE_CROSSCOMPILING)
  # Use opt that is also built in the same project.
  set(opt "$<TARGET_FILE:opt>")
else()
  libomptarget_say("Not building deviceRTL: opt not found")
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

set(devicertl_base_directory ${CMAKE_CURRENT_SOURCE_DIR})
set(include_directory ${devicertl_base_directory}/include)
set(source_directory ${devicertl_base_directory}/src)

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
set(LIBOMPTARGET_DEVICE_DEBUG FALSE CACHE BOOL
  "Activate NVPTX device RTL debug messages.")

if ("${cuda_compiler}" STREQUAL "$<TARGET_FILE:clang>")
  libomptarget_say("Building LLVM bitcode offloading device RTL using in-tree clang.")
else ()
  libomptarget_say("Building LLVM bitcode offloading device RTL using ${cuda_compiler}")
endif ()

set(src_files
  ${source_directory}/Configuration.cpp
  ${source_directory}/Debug.cpp
  ${source_directory}/Kernel.cpp
  ${source_directory}/Mapping.cpp
  ${source_directory}/Misc.cpp
  ${source_directory}/Parallelism.cpp
  ${source_directory}/Reduction.cpp
  ${source_directory}/State.cpp
  ${source_directory}/Synchronization.cpp
  ${source_directory}/Tasking.cpp
  ${source_directory}/Utils.cpp
  ${source_directory}/Workshare.cpp
)

set(clang_opt_flags -O1 -mllvm -openmp-opt-disable -DSHARED_SCRATCHPAD_SIZE=2048)
set(link_opt_flags  -O1        -openmp-opt-disable)

# Prepend -I to each list element
set (LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL "${LIBOMPTARGET_LLVM_INCLUDE_DIRS}")
list(TRANSFORM LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL PREPEND "-I")

# Set flags for LLVM Bitcode compilation.
set(bc_flags -S -x c++ -std=c++17
              ${clang_opt_flags}
             -target nvptx64
             -Xclang -emit-llvm-bc
             -Xclang -aux-triple -Xclang ${aux_triple}
             -fopenmp -fopenmp-cuda-mode -Xclang -fopenmp-is-device
             -Xclang -target-feature -Xclang +ptx61
             -I${include_directory}
             -I${devicertl_base_directory}/../include
             ${LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL}
)

if(${LIBOMPTARGET_DEVICE_DEBUG})
  list(APPEND bc_flags -DOMPTARGET_DEBUG=-1)
else()
  list(APPEND bc_flags -DOMPTARGET_DEBUG=0)
endif()

# Create target to build all Bitcode libraries.
add_custom_target(omptarget-new-nvptx-bc)
add_dependencies(omptarget-new-nvptx-bc opt llvm-link)

# Generate a Bitcode library for all the compute capabilities the user requested
foreach(sm ${nvptx_sm_list})
  # TODO: replace this with declare variant and isa selector.
  set(cuda_flags -Xclang -target-cpu -Xclang sm_${sm} "-D__CUDA_ARCH__=${sm}0")
  set(bc_files "")
  foreach(src ${src_files})
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
    if("${cuda_compiler}" STREQUAL "$<TARGET_FILE:clang>")
      # Add a file-level dependency to ensure that clang is up-to-date.
      # By default, add_custom_command only builds clang if the
      # executable is missing.
      add_custom_command(OUTPUT ${outfile}
        DEPENDS clang
        APPEND
      )
    endif()
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${outfile})

    list(APPEND bc_files ${outfile})
  endforeach()

  set(bclib_name "libomptarget-new-nvptx-sm_${sm}.bc")

  # Link to a bitcode library.
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      COMMAND ${bc_linker}
        -o ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name} ${bc_files}
      DEPENDS ${bc_files}
      COMMENT "Linking LLVM bitcode ${bclib_name}"
  )
  if("${bc_linker}" STREQUAL "$<TARGET_FILE:llvm-link>")
    # Add a file-level dependency to ensure that llvm-link is up-to-date.
    # By default, add_custom_command only builds llvm-link if the
    # executable is missing.
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      DEPENDS llvm-link
      APPEND
    )
  endif()

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}_opt
      COMMAND ${opt} ${link_opt_flags} ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      COMMENT "Optimizing LLVM bitcode ${bclib_name}"
  )
  if("${opt}" STREQUAL "$<TARGET_FILE:opt>")
    # Add a file-level dependency to ensure that opt is up-to-date.
    # By default, add_custom_command only builds opt if the
    # executable is missing.
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}_opt
      DEPENDS opt
      APPEND
    )
  endif()
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${bclib_name})

  set(bclib_target_name "omptarget-new-nvptx-sm_${sm}-bc")

  add_custom_target(${bclib_target_name} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}_opt)
  add_dependencies(omptarget-new-nvptx-bc ${bclib_target_name})
  add_dependencies(${bclib_target_name} opt llvm-link)

  # Copy library to destination.
  add_custom_command(TARGET ${bclib_target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
                    ${LIBOMPTARGET_LIBRARY_DIR})

  # Install bitcode library under the lib destination folder.
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name} DESTINATION "${OPENMP_INSTALL_LIBDIR}")
endforeach()

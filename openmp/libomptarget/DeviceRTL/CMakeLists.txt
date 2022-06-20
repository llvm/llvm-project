##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build the DeviceRTL for all toolchains that are available
#
##===----------------------------------------------------------------------===##

set(LIBOMPTARGET_BUILD_DEVICERTL_BCLIB TRUE CACHE BOOL
  "Can be set to false to disable building this library.")

if (NOT LIBOMPTARGET_BUILD_DEVICERTL_BCLIB)
  libomptarget_say("Not building DeviceRTL: Disabled by LIBOMPTARGET_BUILD_DEVICERTL_BCLIB")
  return()
endif()

if (LLVM_DIR)
  # Builds that use pre-installed LLVM have LLVM_DIR set.
  # A standalone or LLVM_ENABLE_RUNTIMES=openmp build takes this route
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(LINK_TOOL llvm-link PATHS ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)
  find_program(OPT_TOOL opt PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  if ((NOT CLANG_TOOL) OR (NOT LINK_TOOL) OR (NOT OPT_TOOL))
    libomptarget_say("Not building DeviceRTL. Missing clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL} or opt: ${OPT_TOOL}")
    return()
  else()
    libomptarget_say("Building DeviceRTL. Using clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL} and opt: ${OPT_TOOL}")
  endif()
elseif (LLVM_TOOL_CLANG_BUILD AND NOT CMAKE_CROSSCOMPILING AND NOT OPENMP_STANDALONE_BUILD)
  # LLVM in-tree builds may use CMake target names to discover the tools.
  # A LLVM_ENABLE_PROJECTS=openmp build takes this route
  set(CLANG_TOOL $<TARGET_FILE:clang>)
  set(LINK_TOOL $<TARGET_FILE:llvm-link>)
  set(OPT_TOOL $<TARGET_FILE:opt>)
  libomptarget_say("Building DeviceRTL. Using clang from in-tree build")
else()
  libomptarget_say("Not building DeviceRTL. No appropriate clang found")
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
  libomptarget_say("Not building DeviceRTL: unknown host arch: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
  return()
endif()

set(devicertl_base_directory ${CMAKE_CURRENT_SOURCE_DIR})
set(include_directory ${devicertl_base_directory}/include)
set(source_directory ${devicertl_base_directory}/src)

set(all_capabilities 35 37 50 52 53 60 61 62 70 72 75 80 86)

set(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES ${all_capabilities} CACHE STRING
  "List of CUDA Compute Capabilities to be used to compile the NVPTX DeviceRTL.")
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
  libomptarget_say("Not building CUDA offloading DeviceRTL: empty compute capability list")
  return()
endif()

# Check all SM values
foreach(sm ${nvptx_sm_list})
  if (NOT ${sm} IN_LIST all_capabilities)
    libomptarget_warning_say("[NVPTX] Compute capability ${sm} is not supported. Make sure clang can work with it.")
  endif()
endforeach()

set(amdgpu_mcpus gfx700 gfx701 gfx801 gfx803 gfx900 gfx902 gfx906 gfx908 gfx90a gfx90c gfx940 gfx1010 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036)
if (DEFINED LIBOMPTARGET_AMDGCN_GFXLIST)
  set(amdgpu_mcpus ${LIBOMPTARGET_AMDGCN_GFXLIST})
endif()


# Activate RTL message dumps if requested by the user.
set(LIBOMPTARGET_DEVICE_DEBUG FALSE CACHE BOOL
  "Activate DeviceRTL debug messages.")

set(include_files
  ${include_directory}/Configuration.h
  ${include_directory}/Debug.h
  ${include_directory}/Interface.h
  ${include_directory}/Mapping.h
  ${include_directory}/State.h
  ${include_directory}/Synchronization.h
  ${include_directory}/Types.h
  ${include_directory}/Utils.h
)

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

set(clang_opt_flags -O1 -mllvm -openmp-opt-disable -DSHARED_SCRATCHPAD_SIZE=512)
set(link_opt_flags  -O1        -openmp-opt-disable)

# Prepend -I to each list element
set (LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL "${LIBOMPTARGET_LLVM_INCLUDE_DIRS}")
list(TRANSFORM LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL PREPEND "-I")

# Set flags for LLVM Bitcode compilation.
set(bc_flags -S -x c++ -std=c++17 -fvisibility=hidden
              ${clang_opt_flags}
             -Xclang -emit-llvm-bc
             -Xclang -aux-triple -Xclang ${aux_triple}
             -fopenmp -fopenmp-cuda-mode -Xclang -fopenmp-is-device
             -I${include_directory}
             -I${devicertl_base_directory}/../include
             ${LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL}
)

if(${LIBOMPTARGET_DEVICE_DEBUG})
  list(APPEND bc_flags -DOMPTARGET_DEBUG=-1)
else()
  list(APPEND bc_flags -DOMPTARGET_DEBUG=0)
endif()

function(compileDeviceRTLLibrary target_cpu target_name)
  set(target_bc_flags ${ARGN})

  set(bc_files "")
  foreach(src ${src_files})
    get_filename_component(infile ${src} ABSOLUTE)
    get_filename_component(outfile ${src} NAME)
    set(outfile "${outfile}-${target_cpu}.bc")

    add_custom_command(OUTPUT ${outfile}
      COMMAND ${CLANG_TOOL}
      ${bc_flags}
      -Xclang -target-cpu -Xclang ${target_cpu}
      ${target_bc_flags}
      ${infile} -o ${outfile}
      DEPENDS ${infile} ${include_files}
      IMPLICIT_DEPENDS CXX ${infile}
      COMMENT "Building LLVM bitcode ${outfile}"
      VERBATIM
    )
    if("${CLANG_TOOL}" STREQUAL "$<TARGET_FILE:clang>")
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

  set(bclib_name "libomptarget-${target_name}-${target_cpu}.bc")

  # Link to a bitcode library.
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
      COMMAND ${LINK_TOOL}
        -o ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name} ${bc_files}
      DEPENDS ${bc_files}
      COMMENT "Linking LLVM bitcode ${bclib_name}"
  )

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      COMMAND ${OPT_TOOL} ${link_opt_flags} ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
      COMMENT "Optimizing LLVM bitcode ${bclib_name}"
  )

  # Add a file-level dependency to ensure that llvm-link and opt are up-to-date.
  # By default, add_custom_command only builds the tool if the executable is missing
  if("${LINK_TOOL}" STREQUAL "$<TARGET_FILE:llvm-link>")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
      DEPENDS llvm-link
      APPEND)
  endif()
  if("${OPT_TOOL}" STREQUAL "$<TARGET_FILE:opt>")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      DEPENDS opt
      APPEND)
  endif()

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${bclib_name})

  set(bclib_target_name "omptarget-${target_name}-${target_cpu}-bc")

  add_custom_target(${bclib_target_name} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name})

  # Copy library to destination.
  add_custom_command(TARGET ${bclib_target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
                    ${LIBOMPTARGET_LIBRARY_DIR})
  add_dependencies("omptarget.devicertl.${target_name}" "${bclib_target_name}")

  # Install bitcode library under the lib destination folder.
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name} DESTINATION "${OPENMP_INSTALL_LIBDIR}")
endfunction()

# Generate a Bitcode library for all the compute capabilities the user requested
add_custom_target(omptarget.devicertl.nvptx)
foreach(sm ${nvptx_sm_list})
  compileDeviceRTLLibrary(sm_${sm} nvptx -target nvptx64-nvidia-cuda -DLIBOMPTARGET_BC_TARGET -Xclang -target-feature -Xclang +ptx61 "-D__CUDA_ARCH__=${sm}0")
endforeach()

add_custom_target(omptarget.devicertl.amdgpu)
foreach(mcpu ${amdgpu_mcpus})
  compileDeviceRTLLibrary(${mcpu} amdgpu -target amdgcn-amd-amdhsa -DLIBOMPTARGET_BC_TARGET -D__AMDGCN__ -nogpulib)
endforeach()

set(LIBOMPTARGET_LLVM_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang" AND CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL LIBOMPTARGET_LLVM_VERSION))
  libomptarget_say("Not building static library, CMake compiler '${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}' is not 'Clang ${LIBOMPTARGET_LLVM_VERSION}'.")
  libomptarget_say("  Use the 'LLVM_ENABLE_RUNTIMES=openmp' option instead")
  return()
endif()

# Set the flags to build the device runtime from clang.
set(clang_lib_flags -fopenmp -fopenmp-cuda-mode -foffload-lto -fvisibility=hidden -Xopenmp-target=nvptx64-nvidia-cuda --cuda-feature=+ptx61 -mllvm -openmp-opt-disable -nocudalib -nogpulib -nostdinc -DSHARED_SCRATCHPAD_SIZE=512)
foreach(arch ${nvptx_sm_list})
  set(clang_lib_flags ${clang_lib_flags} --offload-arch=sm_${arch})
endforeach()
foreach(arch ${amdgpu_mcpus})
  set(clang_lib_flags ${clang_lib_flags} --offload-arch=${arch})
endforeach()

# Build the static library version of the device runtime.
add_library(omptarget.devicertl STATIC)
target_compile_features(omptarget.devicertl PRIVATE cxx_std_17)
target_include_directories(omptarget.devicertl PRIVATE ${include_directory} ${devicertl_base_directory}/../include)
target_compile_options(omptarget.devicertl PRIVATE ${clang_lib_flags})

install(TARGETS omptarget.devicertl ARCHIVE DESTINATION ${OPENMP_INSTALL_LIBDIR})

add_subdirectory(src)

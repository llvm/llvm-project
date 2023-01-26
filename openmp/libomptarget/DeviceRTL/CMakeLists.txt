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

# Check to ensure the host system is a supported host architecture.
if(NOT ${CMAKE_SIZEOF_VOID_P} EQUAL "8")
  libomptarget_say("Not building DeviceRTL: Runtime does not support 32-bit hosts")
  return()
endif()

if (LLVM_DIR)
  # Builds that use pre-installed LLVM have LLVM_DIR set.
  # A standalone or LLVM_ENABLE_RUNTIMES=openmp build takes this route
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(PACKAGER_TOOL clang-offload-packager PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(LINK_TOOL llvm-link PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(OPT_TOOL opt PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(EXTRACT_TOOL llvm-extract PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  if ((NOT CLANG_TOOL) OR (NOT LINK_TOOL) OR (NOT OPT_TOOL) OR (NOT EXTRACT_TOOL) OR (NOT PACKAGER_TOOL))
    libomptarget_say("Not building DeviceRTL. Missing clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL}, opt: ${OPT_TOOL}, llvm-extract: ${EXTRACT_TOOL}, or clang-offload-packager: ${PACKAGER_TOOL}")
    return()
  else()
    libomptarget_say("Building DeviceRTL. Using clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL} and opt: ${OPT_TOOL}")
  endif()
elseif (LLVM_TOOL_CLANG_BUILD AND NOT CMAKE_CROSSCOMPILING AND NOT OPENMP_STANDALONE_BUILD)
  # LLVM in-tree builds may use CMake target names to discover the tools.
  # A LLVM_ENABLE_PROJECTS=openmp build takes this route
  set(CLANG_TOOL $<TARGET_FILE:clang>)
  set(PACKAGER_TOOL $<TARGET_FILE:clang-offload-packager>)
  set(LINK_TOOL $<TARGET_FILE:llvm-link>)
  set(OPT_TOOL $<TARGET_FILE:opt>)
  set(EXTRACT_TOOL $<TARGET_FILE:llvm-extract>)
  libomptarget_say("Building DeviceRTL. Using clang from in-tree build")
else()
  libomptarget_say("Not building DeviceRTL. No appropriate clang found")
  return()
endif()

set(devicertl_base_directory ${CMAKE_CURRENT_SOURCE_DIR})
set(include_directory ${devicertl_base_directory}/include)
set(source_directory ${devicertl_base_directory}/src)

set(all_capabilities 35 37 50 52 53 60 61 62 70 72 75 80 86 89 90)

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

# Check all SM values
foreach(sm ${nvptx_sm_list})
  if (NOT ${sm} IN_LIST all_capabilities)
    libomptarget_warning_say("[NVPTX] Compute capability ${sm} is not supported. Make sure clang can work with it.")
  endif()
endforeach()

set(amdgpu_mcpus gfx700 gfx701 gfx801 gfx803 gfx900 gfx902 gfx906 gfx908 gfx90a gfx90c gfx940 gfx1010 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103)
if (DEFINED LIBOMPTARGET_AMDGCN_GFXLIST)
  set(amdgpu_mcpus ${LIBOMPTARGET_AMDGCN_GFXLIST})
endif()

set(include_files
  ${include_directory}/Configuration.h
  ${include_directory}/Debug.h
  ${include_directory}/Interface.h
  ${include_directory}/LibC.h
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
  ${source_directory}/LibC.cpp
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

set(clang_opt_flags -O3 -mllvm -openmp-opt-disable -DSHARED_SCRATCHPAD_SIZE=512)
set(link_opt_flags  -O3        -openmp-opt-disable)
set(link_export_flag -passes=internalize -internalize-public-api-file=${source_directory}/exports)
set(link_extract_flag --func='__keep_alive' --delete)

# Prepend -I to each list element
set (LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL "${LIBOMPTARGET_LLVM_INCLUDE_DIRS}")
list(TRANSFORM LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL PREPEND "-I")

# Set flags for LLVM Bitcode compilation.
set(bc_flags -c -emit-llvm -std=c++17 -fvisibility=hidden
              ${clang_opt_flags} --offload-device-only
             -nocudalib -nogpulib -nostdinc
             -fopenmp -fopenmp-cuda-mode
             -I${include_directory}
             -I${devicertl_base_directory}/../include
             ${LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL}
)

# first create an object target
add_library(omptarget.devicertl.all_objs OBJECT IMPORTED)
function(compileDeviceRTLLibrary target_cpu target_name target_triple)
  set(target_bc_flags ${ARGN})

  set(bc_files "")
  foreach(src ${src_files})
    get_filename_component(infile ${src} ABSOLUTE)
    get_filename_component(outfile ${src} NAME)
    set(outfile "${outfile}-${target_cpu}.bc")

    add_custom_command(OUTPUT ${outfile}
      COMMAND ${CLANG_TOOL}
      ${bc_flags}
      --offload-arch=${target_cpu}
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
  if("${LINK_TOOL}" STREQUAL "$<TARGET_FILE:llvm-link>")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
      DEPENDS llvm-link
      APPEND)
  endif()

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/internalized_${bclib_name}
      COMMAND ${OPT_TOOL} ${link_export_flag} ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/internalized_${bclib_name}
      DEPENDS ${source_directory}/exports ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
      COMMENT "Internalizing LLVM bitcode ${bclib_name}"
  )
  if("${OPT_TOOL}" STREQUAL "$<TARGET_FILE:opt>")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/internalized_${bclib_name}
      DEPENDS opt
      APPEND)
  endif()

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      COMMAND ${OPT_TOOL} ${link_opt_flags} ${CMAKE_CURRENT_BINARY_DIR}/internalized_${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/internalized_${bclib_name}
      COMMENT "Optimizing LLVM bitcode ${bclib_name}"
  )
  if("${OPT_TOOL}" STREQUAL "$<TARGET_FILE:opt>")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      DEPENDS opt
      APPEND)
  endif()

  set(bclib_target_name "omptarget-${target_name}-${target_cpu}-bc")
  add_custom_target(${bclib_target_name} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name})

  # Copy library to destination.
  add_custom_command(TARGET ${bclib_target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
                    ${LIBOMPTARGET_LIBRARY_DIR})
  add_dependencies(omptarget.devicertl.${target_name} ${bclib_target_name})

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${bclib_name} ${LIBOMPTARGET_LIBRARY_DIR}/${bclib_name})

  # Install bitcode library under the lib destination folder.
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name} DESTINATION "${OPENMP_INSTALL_LIBDIR}")

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/extracted_${bclib_name}
      COMMAND ${EXTRACT_TOOL} ${link_extract_flag} ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/extracted_${bclib_name}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name} ${bclib_target_name}
      COMMENT "Extracting LLVM bitcode ${bclib_name}"
  )
  if("${EXTRACT_TOOL}" STREQUAL "$<TARGET_FILE:llvm-extract>")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/extracted_${bclib_name}
      DEPENDS llvm-extract
      APPEND)
  endif()

  # Package the bitcode in the bitcode and embed it in an ELF for the static library
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/packaged_${bclib_name}
      COMMAND ${PACKAGER_TOOL} -o ${CMAKE_CURRENT_BINARY_DIR}/packaged_${bclib_name}
        "--image=file=${CMAKE_CURRENT_BINARY_DIR}/extracted_${bclib_name},triple=${target_triple},arch=${target_cpu},kind=openmp"
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/extracted_${bclib_name}
      COMMENT "Packaging LLVM offloading binary ${bclib_name}.out"
  )
  if("${PACKAGER_TOOL}" STREQUAL "$<TARGET_FILE:clang-offload-packager>")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/packaged_${bclib_name}
      DEPENDS clang-offload-packager
      APPEND)
  endif()

  set(output_name "${CMAKE_CURRENT_BINARY_DIR}/devicertl-${target_name}-${target_cpu}.o")
  add_custom_command(OUTPUT ${output_name}
    COMMAND ${CLANG_TOOL} --std=c++17 -c -nostdlib
            -Xclang -fembed-offload-object=${CMAKE_CURRENT_BINARY_DIR}/packaged_${bclib_name}
            -o ${output_name}
            ${source_directory}/Stub.cpp
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/packaged_${bclib_name} ${source_directory}/Stub.cpp
    COMMENT "Embedding LLVM offloading binary in devicertl-${target_name}-${target_cpu}.o"
    VERBATIM
  )
  if("${CLANG_TOOL}" STREQUAL "$<TARGET_FILE:clang>")
    add_custom_command(OUTPUT ${output_name}
      DEPENDS clang
      APPEND)
  endif()

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${output_name})
  set_property(TARGET omptarget.devicertl.all_objs APPEND PROPERTY IMPORTED_OBJECTS ${output_name})
endfunction()

# Generate a Bitcode library for all the compute capabilities the user requested
add_custom_target(omptarget.devicertl.nvptx)
foreach(sm ${nvptx_sm_list})
  compileDeviceRTLLibrary(sm_${sm} nvptx nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda --cuda-feature=+ptx61)
endforeach()

add_custom_target(omptarget.devicertl.amdgpu)
foreach(mcpu ${amdgpu_mcpus})
  compileDeviceRTLLibrary(${mcpu} amdgpu amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa)
endforeach()

# Archive all the object files generated above into a static library
add_library(omptarget.devicertl STATIC)
set_target_properties(omptarget.devicertl PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(omptarget.devicertl PRIVATE omptarget.devicertl.all_objs)

install(TARGETS omptarget.devicertl ARCHIVE DESTINATION ${OPENMP_INSTALL_LIBDIR})

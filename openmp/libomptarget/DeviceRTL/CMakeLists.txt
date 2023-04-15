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
  if ((NOT CLANG_TOOL) OR (NOT LINK_TOOL) OR (NOT OPT_TOOL) OR (NOT EXTRACT_TOOL) OR (NOT_PACKAGER_TOOL))
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

if(LIBOMPTARGET_DEVICE_ARCHITECTURES STREQUAL "all")
  set(LIBOMPTARGET_DEVICE_ARCHITECTURES ${all_gpu_architectures})
elseif(LIBOMPTARGET_DEVICE_ARCHITECTURES STREQUAL "auto" OR
       LIBOMPTARGET_DEVICE_ARCHITECTURES STREQUAL "native")
  if(NOT LIBOMPTARGET_NVPTX_ARCH AND NOT LIBOMPTARGET_AMDGPU_ARCH)
    libomptarget_error_say(
      "Could not find 'amdgpu-arch' and 'nvptx-arch' tools required for 'auto'")
  elseif(NOT LIBOMPTARGET_FOUND_NVIDIA_GPU AND NOT LIBOMPTARGET_FOUND_AMDGPU_GPU)
    libomptarget_error_say("No AMD or NVIDIA GPU found on the system when using 'auto'")
  endif()
  set(nvptx_sm_list ${LIBOMPTARGET_DEP_CUDA_ARCH})
else()
  string(REPLACE "," ";" nvptx_sm_list "${LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES}")
endif()
list(REMOVE_DUPLICATES LIBOMPTARGET_DEVICE_ARCHITECTURES)

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

# Activate RTL message dumps if requested by the user.
set(LIBOMPTARGET_DEVICE_DEBUG FALSE CACHE BOOL
  "Activate DeviceRTL debug messages.")

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
  ${include_directory}/Xteamr.h
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
  ${source_directory}/ExtraMapping.cpp
  ${source_directory}/Xteamr.cpp
  ${source_directory}/Memory.cpp
)

set(clang_opt_flags -O3 -mllvm -openmp-opt-disable -DSHARED_SCRATCHPAD_SIZE=512)
set(link_opt_flags  -O3        -openmp-opt-disable -attributor-enable=module)
set(link_export_flag -passes=internalize -internalize-public-api-file=${source_directory}/exports)
#set(link_extract_flag --func='__keep_alive' --delete)

# Prepend -I to each list element
set (LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL "${LIBOMPTARGET_LLVM_INCLUDE_DIRS}")
list(TRANSFORM LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL PREPEND "-I")

# Set flags for LLVM Bitcode compilation.
set(bc_flags -c -emit-llvm -std=c++17 -fvisibility=hidden
              ${clang_opt_flags} --offload-device-only
             -nocudalib -nogpulib -nostdinc
             -fopenmp -fopenmp-cuda-mode
             -Wno-unknown-cuda-version
             -I${include_directory}
             -I${devicertl_base_directory}/../include
             ${LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL}
)

if(${LIBOMPTARGET_DEVICE_DEBUG})
  list(APPEND bc_flags -DOMPTARGET_DEBUG=-1)
else()
  list(APPEND bc_flags -DOMPTARGET_DEBUG=0)
endif()

# first create an object target
add_library(omptarget.devicertl.all_objs OBJECT IMPORTED)
function(compileDeviceRTLLibrary target_cpu target_name target_triple)
  set(target_bc_flags ${ARGN})

  set(bc_files "")
  foreach(src ${src_files})
    get_filename_component(infile ${src} ABSOLUTE)
    get_filename_component(outfile ${src} NAME)
    set(outfile "${outfile}-${target_cpu}.bc")

    add_custom_target(${outfile}
      COMMAND ${CLANG_TOOL}
      ${bc_flags}
      --offload-arch=${target_cpu}
      ${target_bc_flags}
      ${infile} -o ${outfile}
      DEPENDS ${infile} ${include_files}
      COMMENT "Building LLVM bitcode ${outfile}"
      VERBATIM
    )
    if("${CLANG_TOOL}" STREQUAL "$<TARGET_FILE:clang>")
      # Add a target-level dependency to ensure that clang is up-to-date.
      # By default, add_custom_target only builds clang if the
      # executable is missing.
      add_dependencies(${outfile} clang)
    endif()
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${outfile})

    list(APPEND bc_files ${outfile})
  endforeach()

  set(bclib_name "libomptarget-${target_name}-${target_cpu}.bc")

  # add_custom_command can only depend on local files. "All-inclusive"
  # libomptarget needs bc files from other CMake build directories.
  set(local_depend_files ${bc_files})
  if( ${target_name} STREQUAL "amdgpu" )
    # For amdgpu, add bc libs to make libomptarget bc "all inclusive".
    # During compilation, the libomptarget bc is essentially the only
    # non-user library linked. It is linked once in GPU link phase
    # following llvm-link options: --internalize --only-needed
    if(NOT amd_device_libs_found)
      find_package(AMDDeviceLibs REQUIRED CONFIG
        HINTS
        ${CMAKE_INSTALL_PREFIX}
        PATHS
        /opt/rocm)
      if(AMDDeviceLibs_DIR)
	      set(amd_device_libs_found ON)
        libomptarget_say("DeviceRTLs ${target_cpu}: Getting ROCm device libs from ${AMDDeviceLibs_DIR}")
      else()
        libomptarget_say("DeviceRTLs ${target_cpu}: Not building AMDGCN device RTL: Could not find AMDDeviceLibs package")
        return()
      endif()
    endif()
    get_target_property(ockl_bc_file ockl LOCATION)
    get_target_property(ocml_bc_file ocml LOCATION)
    set(amdgpu_wfsz_is32 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103)
    string(FIND "${amdgpu_wfsz_is32}" "${target_cpu}" is_32bit)
    if(NOT is_32bit EQUAL -1)
      get_target_property(oclc_wf_bc_file oclc_wavefrontsize64_off LOCATION)
    else()
      get_target_property(oclc_wf_bc_file oclc_wavefrontsize64_on LOCATION)
    endif()
    string(LENGTH "${target_cpu}" gfxlen)
    if(gfxlen EQUAL 6)
      string(SUBSTRING ${target_cpu} 3 3 gfxnum)
    else()
      string(SUBSTRING ${target_cpu} 3 4 gfxnum)
    endif()
    get_target_property(oclc_isa_bc_file oclc_isa_version_${gfxnum} LOCATION)

    # Add custom target so targets from other directories
    # can be added as dependencies to ensure libm
    # and libhostexec bc files have been built.
    add_custom_target(touch-target-${target_cpu} ALL)
    add_dependencies(touch-target-${target_cpu}
      libm-target-${target_cpu}
      libhostexec-amdgcn-${target_cpu}.bc
      #${CMAKE_BINARY_DIR}/libomptarget/hostexec/libhostexec-amdgcn-${target_cpu}.bc
    )
    list(APPEND bc_files
      ${CMAKE_BINARY_DIR}/libm-amdgcn-${target_cpu}.bc
      ${CMAKE_BINARY_DIR}/libomptarget/hostexec/libhostexec-amdgcn-${target_cpu}.bc
    )
    if (EXISTS ${CMAKE_BINARY_DIR}/../../tools/ROCMDEVLIBS)
      add_dependencies(touch-target-${target_cpu}
        ockl ocml oclc_wavefrontsize64_on oclc_wavefrontsize64_off oclc_isa_version_${gfxnum})
    endif()
    # Add amdgcn-specific bc files to link command
    list(APPEND bc_files ${ocml_bc_file} ${ockl_bc_file} ${oclc_wf_bc_file} ${oclc_isa_bc_file} ${oclc_abi_version_file})
    # Add touch-target-$(target_cpu) to local_depend_files so all $bc_files will exist or be created.
    list(APPEND local_depend_files touch-target-${target_cpu})
  endif()

  # Link to a bitcode library.
  add_custom_target(linked_${bclib_name}
      COMMAND ${LINK_TOOL}
        -o ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name} ${bc_files}
      COMMENT "Linking LLVM bitcode ${bclib_name}"
  )
  add_dependencies(linked_${bclib_name} ${local_depend_files})

  add_custom_target(internalized_${bclib_name}
      COMMAND ${OPT_TOOL} ${link_export_flag} ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/internalized_${bclib_name}
      DEPENDS ${source_directory}/exports
      COMMENT "Internalizing LLVM bitcode ${bclib_name}"
  )
  add_dependencies(internalized_${bclib_name} linked_${bclib_name})

  add_custom_target(${bclib_name}
      COMMAND ${OPT_TOOL} ${link_opt_flags} ${CMAKE_CURRENT_BINARY_DIR}/internalized_${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      COMMENT "Optimizing LLVM bitcode ${bclib_name}"
  )
  add_dependencies(${bclib_name} internalized_${bclib_name})

  add_custom_target(extracted_${bclib_name}
      COMMAND ${EXTRACT_TOOL} ${link_extract_flag} ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name} -o ${CMAKE_CURRENT_BINARY_DIR}/extracted_${bclib_name}
      COMMENT "Extracting LLVM bitcode ${bclib_name}"
  )
  add_dependencies(extracted_${bclib_name} ${bclib_name})

  # Package the bitcode in the bitcode and embed it in an ELF for the static library
  add_custom_target(packaged_${bclib_name}
      COMMAND ${PACKAGER_TOOL} -o ${CMAKE_CURRENT_BINARY_DIR}/packaged_${bclib_name}
        "--image=file=${CMAKE_CURRENT_BINARY_DIR}/${bclib_name},triple=${target_triple},arch=${target_cpu},kind=openmp"
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      COMMENT "Packaging LLVM offloading binary ${bclib_name}.out"
  )
  add_dependencies(packaged_${bclib_name} ${bclib_name})

  set(output_name "${CMAKE_CURRENT_BINARY_DIR}/devicertl-${target_name}-${target_cpu}.o")
  add_custom_target(embedded_${target_name}-${target_cpu}
    COMMAND ${CLANG_TOOL} --std=c++17 -c -nostdlib
            -Xclang -fembed-offload-object=${CMAKE_CURRENT_BINARY_DIR}/packaged_${bclib_name}
            -o ${output_name}
            ${source_directory}/Stub.cpp
    BYPRODUCTS ${output_name}
    COMMENT "Embedding LLVM offloading binary in ${output_name}"
    VERBATIM
  )
  add_dependencies(embedded_${target_name}-${target_cpu} packaged_${bclib_name})

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${output_name})
  add_dependencies(omptarget.devicertl.all_objs embedded_${target_name}-${target_cpu})
  set_property(TARGET omptarget.devicertl.all_objs APPEND PROPERTY IMPORTED_OBJECTS ${output_name})

  # Add a target-level dependency to ensure that llvm-link and opt are up-to-date.
  # By default, add_custom_target only builds the tool if the executable is missing
  if("${LINK_TOOL}" STREQUAL "$<TARGET_FILE:llvm-link>")
    add_dependencies(linked_${bclib_name} llvm-link)
  endif()
  if("${OPT_TOOL}" STREQUAL "$<TARGET_FILE:opt>")
    add_dependencies(${bclib_name} opt)
  endif()
  if("${EXTRACT_TOOL}" STREQUAL "$<TARGET_FILE:llvm-extract>")
    add_dependencies(extracted_${bclib_name} llvm-extract)
  endif()
  if("${PACKAGER_TOOL}" STREQUAL "$<TARGET_FILE:clang-offload-packager>")
    add_dependencies(packaged_${bclib_name} clang-offload-packager)
  endif()
  if("${CLANG_TOOL}" STREQUAL "$<TARGET_FILE:clang>")
    add_dependencies(embedded_${target_name}-${target_cpu} clang)
  endif()

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${bclib_name})

  set(bclib_target_name "omptarget-${target_name}-${target_cpu}-bc")

  add_custom_target(${bclib_target_name} ALL)
  add_dependencies(${bclib_target_name} ${bclib_name})

  # Copy library to destination.
  add_custom_command(TARGET ${bclib_target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
                    ${LIBOMPTARGET_LIBRARY_DIR})
  add_dependencies("omptarget.devicertl.${target_name}" "${bclib_target_name}")

  # _____________________________________________________________________________
  # ------  SUPPORT FOR OLD DRIVER bc files
  if( ${target_name} STREQUAL "amdgpu" )
    set(bclib_name_old_driver "libomptarget-old-${target_name}-${target_cpu}.bc")
    add_custom_target(opt_${bclib_name_old_driver}
      COMMAND ${OPT_TOOL} ${link_opt_flags} ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/opt_${bclib_name_old_driver}
      DEPENDS ${source_directory}/exports
      COMMENT "Generating opt_${bclib_name_old_driver}"
    )
    add_dependencies(opt_${bclib_name_old_driver} linked_${bclib_name})
    add_custom_target(${bclib_name_old_driver}
      COMMAND ${PREP_TOOL} ${CMAKE_CURRENT_BINARY_DIR}/opt_${bclib_name_old_driver}
                      -o ${bclib_name_old_driver}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/opt_${bclib_name_old_driver}
      COMMENT "Running ${PREP_TOOL} for ${bclib_name_old_driver}"
    )
    add_dependencies(${bclib_name_old_driver} opt_${bclib_name_old_driver})
    if("${PREP_TOOL}" STREQUAL "$<TARGET_FILE:prep-libomptarget-bc>")
      add_dependencies(${bclib_name_old_driver} prep-libomptarget-bc)
    endif()
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${bclib_name_old_driver})
    set(bclib_target_name_old "omptarget-old-${target_name}-${target_cpu}-bc")
    add_custom_target(${bclib_target_name_old} ALL)
    add_dependencies(${bclib_target_name_old} ${bclib_name_old_driver})
    add_custom_command(TARGET ${bclib_target_name_old} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name_old_driver}
                    ${LIBOMPTARGET_LIBRARY_DIR})
    add_dependencies("omptarget.devicertl.${target_name}" "${bclib_target_name_old}")
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name_old_driver} DESTINATION "${DEVEL_PACKAGE}${OPENMP_INSTALL_LIBDIR}")
  endif()
# _____________________________________________________________________________
endfunction()

# Generate a Bitcode library for all the compute capabilities the user requested
add_custom_target(omptarget.devicertl.nvptx)
foreach(sm ${nvptx_sm_list})
  compileDeviceRTLLibrary(sm_${sm} nvptx nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_${sm} -DLIBOMPTARGET_BC_TARGET --cuda-feature=+ptx61)
endforeach()

add_custom_target(omptarget.devicertl.amdgpu)
foreach(mcpu ${amdgpu_mcpus})
  compileDeviceRTLLibrary(${mcpu} amdgpu amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${mcpu} -DLIBOMPTARGET_BC_TARGET -D__AMDGCN__ -nogpulib)
endforeach()

# Archive all the object files generated above into a static library
add_library(omptarget.devicertl STATIC)
set_target_properties(omptarget.devicertl PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(omptarget.devicertl PRIVATE omptarget.devicertl.all_objs)

install(TARGETS omptarget.devicertl ARCHIVE DESTINATION ${DEVEL_PACKAGE}${OPENMP_INSTALL_LIBDIR})

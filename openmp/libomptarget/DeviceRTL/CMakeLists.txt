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
  if ((NOT CLANG_TOOL) OR (NOT LINK_TOOL) OR (NOT OPT_TOOL) OR (NOT PACKAGER_TOOL))
    libomptarget_say("Not building DeviceRTL. Missing clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL}, opt: ${OPT_TOOL}, or clang-offload-packager: ${PACKAGER_TOOL}")

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
  libomptarget_say("Building DeviceRTL. Using clang from in-tree build")
else()
  libomptarget_say("Not building DeviceRTL. No appropriate clang found")
  return()
endif()

set(devicertl_base_directory ${CMAKE_CURRENT_SOURCE_DIR})
set(include_directory ${devicertl_base_directory}/include)
set(source_directory ${devicertl_base_directory}/src)

set(all_capabilities 35 37 50 52 53 60 61 62 70 72 75 80 86 89 87 90)
set(all_amdgpu_architectures "gfx700;gfx701;gfx801;gfx803;gfx900;gfx902;gfx906"
                             "gfx908;gfx90a;gfx90c;gfx940;gfx941;gfx942;gfx1010"
                             "gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035"
                             "gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150"
                             "gfx1151")
set(all_nvptx_architectures "sm_35;sm_37;sm_50;sm_52;sm_53;sm_60;sm_61;sm_62"
                            "sm_70;sm_72;sm_75;sm_80;sm_86;sm_87;sm_89;sm_90")
set(all_gpu_architectures
    "${all_amdgpu_architectures};${all_nvptx_architectures}")

# AMD internal build scripts use LIBOMPTARGET_AMDGCN_GFXLIST
if(DEFINED LIBOMPTARGET_AMDGCN_GFXLIST)
  set(LIBOMPTARGET_DEVICE_ARCHITECTURES ${LIBOMPTARGET_AMDGCN_GFXLIST} CACHE STRING
      "List of device architectures to be used to compile the OpenMP DeviceRTL.")
else()
  set(LIBOMPTARGET_DEVICE_ARCHITECTURES "all" CACHE STRING
      "List of device architectures to be used to compile the OpenMP DeviceRTL.")
endif()

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
  set(LIBOMPTARGET_DEVICE_ARCHITECTURES
      "${LIBOMPTARGET_NVPTX_DETECTED_ARCH_LIST};${LIBOMPTARGET_AMDGPU_DETECTED_ARCH_LIST}")
endif()
list(REMOVE_DUPLICATES LIBOMPTARGET_DEVICE_ARCHITECTURES)

set(include_files
  ${include_directory}/Allocator.h
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
  ${include_directory}/Xteams.h
)

set(src_files
  ${source_directory}/Allocator.cpp
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
  ${source_directory}/Xteams.cpp
)

# We disable the slp vectorizer during the runtime optimization to avoid
# vectorized accesses to the shared state. Generally, those are "good" but
# the optimizer pipeline (esp. Attributor) does not fully support vectorized
# instructions yet and we end up missing out on way more important constant
# propagation. That said, we will run the vectorizer again after the runtime 
# has been linked into the user program.
set(clang_opt_flags -O3 -mllvm -openmp-opt-disable -DSHARED_SCRATCHPAD_SIZE=512 -mllvm -vectorize-slp=false )
set(link_opt_flags  -O3        -openmp-opt-disable -attributor-enable=module -vectorize-slp=false )
set(link_export_flag -passes=internalize -internalize-public-api-file=${source_directory}/exports)

# Prepend -I to each list element
set (LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL "${LIBOMPTARGET_LLVM_INCLUDE_DIRS}")
list(TRANSFORM LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL PREPEND "-I")

# Set flags for LLVM Bitcode compilation.
set(bc_flags -c -foffload-lto -std=c++17 -fvisibility=hidden
              ${clang_opt_flags} --offload-device-only
             -nocudalib -nogpulib -nostdinc
             -fopenmp -fopenmp-cuda-mode
             -Wno-unknown-cuda-version
             -DOMPTARGET_DEVICE_RUNTIME
             -I${include_directory}
             -I${devicertl_base_directory}/../include
             ${LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL}
)

if(${LIBOMPTARGET_DEVICE_DEBUG})
  list(APPEND bc_flags -DOMPTARGET_DEBUG=-1)
else()
  list(APPEND bc_flags -DOMPTARGET_DEBUG=0)
endif()

function(addAMDSpecificBcLibs touch_target gfxname bc_files local_depend_files)
  # For amdgpu, the libomptarget bc is "all inclusive".
  # During user compilation, the libomptarget bc is essentially the only
  # non-user library linked. It is linked once in GPU link phase
  # following llvm-link options: --internalize --only-needed
  if(NOT amd_device_libs_found)
    find_package(AMDDeviceLibs REQUIRED CONFIG
      HINTS
      ${CMAKE_BINARY_DIR}/../../tools/rocm-device-libs
      ${CMAKE_INSTALL_PREFIX}
      PATHS
      /opt/rocm)
    if(AMDDeviceLibs_DIR)
      set(amd_device_libs_found ON)
      libomptarget_say("DeviceRTLs ${gfxname}: Getting ROCm device libs from ${AMDDeviceLibs_DIR}")
    else()
      libomptarget_say("DeviceRTLs ${gfxname}: Not building AMDGCN device RTL: Could not find AMDDeviceLibs package")
      return()
    endif()
  endif()
  get_target_property(ockl_bc_file ockl LOCATION)
  get_target_property(ocml_bc_file ocml LOCATION)
  set(amdgpu_wfsz_is32 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103)
  string(FIND "${amdgpu_wfsz_is32}" "${gfxname}" is_32bit)
  if(NOT is_32bit EQUAL -1)
    get_target_property(oclc_wf_bc_file oclc_wavefrontsize64_off LOCATION)
  else()
    get_target_property(oclc_wf_bc_file oclc_wavefrontsize64_on LOCATION)
  endif()
  string(LENGTH "${gfxname}" gfxlen)
  if(gfxlen EQUAL 6)
    string(SUBSTRING ${gfxname} 3 3 gfxnum)
  else()
    string(SUBSTRING ${gfxname} 3 4 gfxnum)
  endif()
  get_target_property(oclc_isa_bc_file oclc_isa_version_${gfxnum} LOCATION)

  # Add custom target so targets from other directories
  # can be added as dependencies to ensure libm
  # and libhostexec bc files have been built.
  add_custom_target(${touch_target} ALL)
  add_dependencies(${touch_target}
    libm-target-${gfxname}
    libhostexec-${gfxname}.bc
  )

  # TODO: Add back -amdgpu to the names below (maybe?).
  list(APPEND bc_files
    ${CMAKE_BINARY_DIR}/libm-${gfxname}.bc
  #  ${CMAKE_BINARY_DIR}/openmp/libomptarget/hostexec/libhostexec-${gfxname}.bc
  )
  if(OPENMP_STANDALONE_BUILD)
    list(APPEND bc_files
      ${CMAKE_BINARY_DIR}/libomptarget/hostexec/libhostexec-${gfxname}.bc
    )
  else()
    list(APPEND bc_files
      ${CMAKE_BINARY_DIR}/openmp/libomptarget/hostexec/libhostexec-${gfxname}.bc
    )
  endif()
  if (EXISTS ${CMAKE_BINARY_DIR}/../../tools/ROCMDEVLIBS)
    add_dependencies(${touch_target}
      ockl ocml oclc_wavefrontsize64_on oclc_wavefrontsize64_off oclc_isa_version_${gfxnum})
  endif()

  # Add amdgcn-specific bc files to link command
  list(APPEND bc_files ${ocml_bc_file} ${ockl_bc_file} ${oclc_wf_bc_file} ${oclc_isa_bc_file})

  # Add touch-target-$(target_cpu) to local_depend_files so all $bc_files will exist or be created.
  list(APPEND local_depend_files ${touch_target})

  # Update these values in the caller:
  set (bc_files ${bc_files} PARENT_SCOPE)
  set (local_depend_files ${local_depend_files} PARENT_SCOPE)
endfunction()

# first create an object target
add_library(omptarget.devicertl.all_objs OBJECT IMPORTED)
function(compileDeviceRTLLibrary target_cpu target_name target_triple)
  set(target_bc_flags ${ARGN})

  set(bc_files "")
  foreach(src ${src_files})
    get_filename_component(infile ${src} ABSOLUTE)
    get_filename_component(outfile ${src} NAME)
    set(outfile "${outfile}-${target_cpu}.bc")
    set(depfile "${outfile}.d")

    add_custom_target(${outfile}
      COMMAND ${CLANG_TOOL}
      ${bc_flags}
      --offload-arch=${target_cpu}
      ${target_bc_flags}
      -MD -MF ${depfile}
      ${infile} -o ${outfile}
      DEPENDS ${infile} ${include_files}
      COMMENT "Building LLVM bitcode ${outfile}"
      VERBATIM
    )
    if("${CLANG_TOOL}" STREQUAL "$<TARGET_FILE:clang>")
      # Add a file-level dependency to ensure that clang is up-to-date.
      # By default, add_custom_command only builds clang if the
      # executable is missing.
      add_dependencies(${outfile} clang)
    endif()
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${outfile})

    list(APPEND bc_files ${outfile})
  endforeach()

  set(bclib_name "libomptarget-${target_name}-${target_cpu}.bc")

  set(local_depend_files ${bc_files})
  if( ${target_name} STREQUAL "amdgpu" )
    addAMDSpecificBcLibs("touch-target-${target_cpu}" ${target_cpu} "${bc_files}" "${local_depend_files}")
  endif()

  # Link to a bitcode library.
  add_custom_target(linked_${bclib_name}
      COMMAND ${LINK_TOOL}
        -o ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name} ${bc_files}
      DEPENDS ${bc_files}
      COMMENT "Linking LLVM bitcode ${bclib_name}"
  )
  add_dependencies(linked_${bclib_name} ${local_depend_files})
  if("${LINK_TOOL}" STREQUAL "$<TARGET_FILE:llvm-link>")
    add_dependencies(linked_${bclib_name} llvm-link)
  endif()

  add_custom_target(internalized_${bclib_name}
      COMMAND ${OPT_TOOL} ${link_export_flag} ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/internalized_${bclib_name}
      DEPENDS ${source_directory}/exports
      COMMENT "Internalizing LLVM bitcode ${bclib_name}"
  )
  add_dependencies(internalized_${bclib_name} linked_${bclib_name})
  if("${OPT_TOOL}" STREQUAL "$<TARGET_FILE:opt>")
    add_dependencies(internalized_${bclib_name} opt)
  endif()

  add_custom_target(opt_${bclib_name}
      COMMAND ${OPT_TOOL} ${link_opt_flags} ${CMAKE_CURRENT_BINARY_DIR}/internalized_${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/opt_${bclib_name}
      COMMENT "Optimizing LLVM bitcode opt_${bclib_name}"
  )
  add_dependencies(opt_${bclib_name} internalized_${bclib_name})
  if("${OPT_TOOL}" STREQUAL "$<TARGET_FILE:opt>")
    add_dependencies(opt_${bclib_name} opt)
  endif()

  #  ----------------- steps for bc file installation for opaque-linker ----------------- #

  add_custom_target(preprep_${bclib_name}
      COMMAND ${OPT_TOOL} ${link_opt_flags} ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/preprep_${bclib_name}
      COMMENT "Optimizing LLVM bitcode preprep_${bclib_name}"
  )
  add_dependencies(preprep_${bclib_name} linked_${bclib_name})
  if("${OPT_TOOL}" STREQUAL "$<TARGET_FILE:opt>")
    add_dependencies(preprep_${bclib_name} opt)
  endif()

  #  Run the prep tool on the opt library to create the installable bc file used in opaque-linker
  add_custom_target(${bclib_name}
      COMMAND ${PREP_TOOL} ${CMAKE_CURRENT_BINARY_DIR}/preprep_${bclib_name}
                      -o ${bclib_name}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/preprep_${bclib_name}
      COMMENT "Running ${PREP_TOOL} for ${bclib_name}"
  )
  add_dependencies(${bclib_name} preprep_${bclib_name})
  if("${PREP_TOOL}" STREQUAL "$<TARGET_FILE:prep-libomptarget-bc>")
    add_dependencies(${bclib_name} prep-libomptarget-bc)
  endif()

  add_dependencies(omptarget.devicertl.all_objs embedded_${target_name}-${target_cpu})

  set(bclib_target_name "omptarget-${target_name}-${target_cpu}-bc")
  add_custom_target(${bclib_target_name} ALL)
  add_dependencies(${bclib_target_name} ${bclib_name})

  # Copy library to destination.
  add_custom_command(TARGET ${bclib_target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
                    ${LIBOMPTARGET_LIBRARY_DIR})
  add_dependencies(omptarget.devicertl.${target_name} ${bclib_target_name})
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES  ${bclib_name} ${LIBOMPTARGET_LIBRARY_DIR}/${bclib_name})

  # Install bitcode library under the lib destination folder.
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name} DESTINATION "${DEVEL_PACKAGE}${OPENMP_INSTALL_LIBDIR}")

  #  ----------------- end steps for bc file installation for opaque-linker ----------------- #

  # TODO: The NVPTX target feature needs to be added before the packaging
  # The target_feature is optional. The comma and the feature get added together.
  set(target_feature "")
  if("${target_triple}" STREQUAL "nvptx64-nvidia-cuda")
    set(target_feature "feature=+ptx63")
  endif()

  # Package the bitcode in the bitcode and embed it in an ELF for the static library
  add_custom_target(packaged_${bclib_name}
      COMMAND ${PACKAGER_TOOL} -o ${CMAKE_CURRENT_BINARY_DIR}/packaged_${bclib_name}
        "--image=file=${CMAKE_CURRENT_BINARY_DIR}/opt_${bclib_name},${target_feature},triple=${target_triple},arch=${target_cpu},kind=openmp"
      COMMENT "Packaging LLVM offloading binary ${bclib_name}.out"
  )
  add_dependencies(packaged_${bclib_name} opt_${bclib_name})
  if("${PACKAGER_TOOL}" STREQUAL "$<TARGET_FILE:clang-offload-packager>")
    add_dependencies(packaged_${bclib_name} clang-offload-packager)
  endif()

  set(output_name "${CMAKE_CURRENT_BINARY_DIR}/devicertl-${target_name}-${target_cpu}.o")
  add_custom_target(embedded_${target_name}-${target_cpu}
    COMMAND ${CLANG_TOOL} --std=c++17 -c -nostdlib
            -Xclang -fembed-offload-object=${CMAKE_CURRENT_BINARY_DIR}/packaged_${bclib_name}
            -o ${output_name}
            ${source_directory}/Stub.cpp
    DEPENDS ${source_directory}/Stub.cpp
    BYPRODUCTS ${output_name}
    COMMENT "Embedding LLVM offloading binary in devicertl-${target_name}-${target_cpu}.o"
    VERBATIM
  )
  add_dependencies(embedded_${target_name}-${target_cpu} packaged_${bclib_name})
  if("${CLANG_TOOL}" STREQUAL "$<TARGET_FILE:clang>")
    add_dependencies(${output_name} clang)
  endif()

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${output_name})
  set_property(TARGET omptarget.devicertl.all_objs APPEND PROPERTY IMPORTED_OBJECTS ${output_name})

  if (CMAKE_EXPORT_COMPILE_COMMANDS)
    set(ide_target_name omptarget-ide-${target_name}-${target_cpu})
    add_library(${ide_target_name} STATIC EXCLUDE_FROM_ALL ${src_files})
    target_compile_options(${ide_target_name} PRIVATE
      -fopenmp --offload-arch=${target_cpu} -fopenmp-cuda-mode
      -mllvm -openmp-opt-disable
      -foffload-lto -fvisibility=hidden --offload-device-only
      -nocudalib -nogpulib -nostdinc -Wno-unknown-cuda-version
    )
    target_compile_definitions(${ide_target_name} PRIVATE SHARED_SCRATCHPAD_SIZE=512)
    target_include_directories(${ide_target_name} PRIVATE
      ${include_directory}
      ${devicertl_base_directory}/../include
      ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
    )
    install(TARGETS ${ide_target_name} EXCLUDE_FROM_ALL)
  endif()
endfunction()

# Generate a Bitcode library for all the gpu architectures the user requested.
add_custom_target(omptarget.devicertl.nvptx)
add_custom_target(omptarget.devicertl.amdgpu)
foreach(gpu_arch ${LIBOMPTARGET_DEVICE_ARCHITECTURES})
  if("${gpu_arch}" IN_LIST all_amdgpu_architectures)
    set(clang_options -DLIBOMPTARGET_BC_TARGET -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${gpu_arch} -DLIBOMPTARGET_BC_TARGET -D__AMDGCN__ -nogpulib)
    compileDeviceRTLLibrary(${gpu_arch} amdgpu amdgcn-amd-amdhsa -Xclang -mcode-object-version=none ${clang_options})
  elseif("${gpu_arch}" IN_LIST all_nvptx_architectures)
    compileDeviceRTLLibrary(${gpu_arch} nvptx nvptx64-nvidia-cuda --cuda-feature=+ptx61)
  else()
    libomptarget_error_say("Unknown GPU architecture '${gpu_arch}'")
  endif()
endforeach()

# Archive all the object files generated above into a static library
add_library(omptarget.devicertl STATIC)
set_target_properties(omptarget.devicertl PROPERTIES
# this break lib-debug and lib-perf builds
# ARCHIVE_OUTPUT_DIRECTORY ${LIBOMPTARGET_LLVM_LIBRARY_INTDIR}
  LINKER_LANGUAGE CXX
)
target_link_libraries(omptarget.devicertl PRIVATE omptarget.devicertl.all_objs)

install(TARGETS omptarget.devicertl ARCHIVE DESTINATION "${DEVEL_PACKAGE}${OPENMP_INSTALL_LIBDIR}")

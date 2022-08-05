##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build the AMDGCN Device RTL bitcode library using clang -ffreestanding
#
##===----------------------------------------------------------------------===##

if ( WIN32 )
  libomptarget_say("Not buildingAMDGCN deviceRTL on windows")
  return()
endif()
find_package(AMDDeviceLibs REQUIRED CONFIG PATHS
	${CMAKE_BINARY_DIR}/../../tools/ROCMDEVLIBS
	${CMAKE_INSTALL_PREFIX}
	/opt/rocm)
if(AMDDeviceLibs_DIR)
  libomptarget_say("Getting ROCm device libs from ${AMDDeviceLibs_DIR}")
else()
  libomptarget_say("Not building AMDGCN device RTL: Could not find AMDDeviceLibs package")
  return()
endif()

set(LIBOMPTARGET_BUILD_AMDGCN_BCLIB TRUE CACHE BOOL
  "Can be set to false to disable building this library.")

if (NOT LIBOMPTARGET_BUILD_AMDGCN_BCLIB)
  libomptarget_say("Not building AMDGCN device RTL: Disabled by LIBOMPTARGET_BUILD_AMDGCN_BCLIB")
  return()
endif()

if (NOT LIBOMPTARGET_LLVM_INCLUDE_DIRS)
  libomptarget_say("Not building AMDGCN device RTL: Missing definition for LIBOMPTARGET_LLVM_INCLUDE_DIRS")
  return()
endif()

if (LLVM_DIR)
  libomptarget_say("Building AMDGCN device RTLs. Using clang: ${CLANG_TOOL}")
elseif (LLVM_TOOL_CLANG_BUILD AND NOT CMAKE_CROSSCOMPILING AND NOT OPENMP_STANDALONE_BUILD)
  # LLVM in-tree builds may use CMake target names to discover the tools.
  libomptarget_say("Building AMDGCN device RTL. Using clang from in-tree build: ${CLANG_TOOL}")
else()
  libomptarget_say("Not building AMDGCN device RTL. No appropriate clang found")
  return()
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(aux_triple x86_64-unknown-linux-gnu)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "ppc64le")
  set(aux_triple powerpc64le-unknown-linux-gnu)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(aux_triple aarch64-unknown-linux-gnu)
else()
  libomptarget_say("Not building AMDGCN device RTL: unknown host arch: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
  return()
endif()

if (LLVM_DIR)
  # Builds that use pre-installed LLVM have LLVM_DIR set.
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(LINK_TOOL llvm-link PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(OPT_TOOL opt PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  if ((NOT CLANG_TOOL) OR (NOT LINK_TOOL) OR (NOT OPT_TOOL))
    libomptarget_say("Not building AMDGCN device RTL. Missing clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL} or opt: ${OPT_TOOL}")
    return()
  else()
    libomptarget_say("Building AMDGCN device RTL. Using clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL} and opt: ${OPT_TOOL}")
  endif()
elseif (LLVM_TOOL_CLANG_BUILD AND NOT CMAKE_CROSSCOMPILING AND NOT OPENMP_STANDALONE_BUILD)
  # LLVM in-tree builds may use CMake target names to discover the tools.
  set(CLANG_TOOL $<TARGET_FILE:clang>)
  set(LINK_TOOL $<TARGET_FILE:llvm-link>)
  set(OPT_TOOL $<TARGET_FILE:opt>)
  libomptarget_say("Building AMDGCN device RTL. Using clang from in-tree build")
else()
  libomptarget_say("Not building AMDGCN device RTL. No appropriate clang found")
  return()
endif()

project(omptarget-amdgcn)

add_custom_target(omptarget-amdgcn ALL)

#optimization level
set(optimization_level 2)
set(clang_opt_flags -O${optimization_level} -mllvm -openmp-opt-disable -DSHARED_SCRATCHPAD_SIZE=2048)
set(link_opt_flags -O${optimization_level} -openmp-opt-disable)

# Activate RTL message dumps if requested by the user.
if(LIBOMPTARGET_NVPTX_DEBUG)
  set(CUDA_DEBUG -DOMPTARGET_NVPTX_DEBUG=-1 -g)
endif()

get_filename_component(devicertl_base_directory
  ${CMAKE_CURRENT_SOURCE_DIR}
  DIRECTORY)

set(hip_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/amdgcn_libcall.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/src/amdgcn_locks.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/src/amdgcn_smid.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/src/target_impl.hip
  ${devicertl_base_directory}/common/src/cancel.cu
  ${devicertl_base_directory}/common/src/critical.cu
  ${devicertl_base_directory}/common/src/data_sharing.cu
  ${devicertl_base_directory}/common/src/libcall.cu
  ${devicertl_base_directory}/common/src/loop.cu
  ${devicertl_base_directory}/common/src/omp_data.cu
  ${devicertl_base_directory}/common/src/omptarget.cu
  ${devicertl_base_directory}/common/src/parallel.cu
  ${devicertl_base_directory}/common/src/reduction.cu
  ${devicertl_base_directory}/common/src/support.cu
  ${devicertl_base_directory}/common/src/shuffle.cpp
  ${devicertl_base_directory}/common/src/sync.cu
  ${devicertl_base_directory}/common/src/task.cu
  ${devicertl_base_directory}/common/src/ompd-specific.cu)

set(h_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/amdgcn_interface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/target_impl.h
  ${devicertl_base_directory}/common/debug.h
  ${devicertl_base_directory}/common/omptarget.h
  ${devicertl_base_directory}/common/omptargeti.h
  ${devicertl_base_directory}/common/state-queue.h
  ${devicertl_base_directory}/common/state-queuei.h
  ${devicertl_base_directory}/common/support.h)

# create gfx bitcode libraries
set(amdgpu_mcpus gfx700 gfx701 gfx801 gfx803 gfx900 gfx902 gfx906 gfx908 gfx90a gfx90c gfx940 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036)
if (DEFINED LIBOMPTARGET_AMDGCN_GFXLIST)
  set(amdgpu_mcpus ${LIBOMPTARGET_AMDGCN_GFXLIST})
endif()
set(amdgpu_32bitwf gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036)

# see if this build is for LLVM_ENABLE_RUNTIMES='openmp'
set(_xdir "")
foreach(proj ${LLVM_ENABLE_RUNTIMES})
  string(TOUPPER "${proj}" canon_name)
  if ("${canon_name}" STREQUAL "OPENMP")
     set(_xdir "/openmp")
  endif()
endforeach()

# Prepend -I to each list element
set (LIBOMPTARGET_LLVM_INCLUDE_DIRS_AMDGCN "${LIBOMPTARGET_LLVM_INCLUDE_DIRS}")
list(TRANSFORM LIBOMPTARGET_LLVM_INCLUDE_DIRS_AMDGCN PREPEND "-I")

macro(build_bc_library mcpu target_libname)
  set(clang_cmd ${CLANG_TOOL}
    -xc++
    -c
    -std=c++14
    -target amdgcn-amd-amdhsa
    -ffreestanding
    -emit-llvm
    -Xclang -aux-triple -Xclang ${aux_triple}
    -fopenmp -fopenmp-cuda-mode -Xclang -fopenmp-is-device
    -D__AMDGCN__
    -Xclang -target-cpu -Xclang ${mcpu}
    -fvisibility=hidden
    -Wno-unused-value
    -nogpulib
    ${clang_opt_flags}
    ${CUDA_DEBUG}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${devicertl_base_directory}/common/include
    -I${devicertl_base_directory}
    -I${devicertl_base_directory}/../include
    ${LIBOMPTARGET_LLVM_INCLUDE_DIRS_AMDGCN})

  set(source_bc_files)

  foreach(file ${hip_sources})
    get_filename_component(fname ${file} NAME_WE)
    set(source_bc_file ${fname}.${mcpu}.bc)

    add_custom_command(
      OUTPUT ${source_bc_file}
      COMMAND ${clang_cmd} ${file} -o ${source_bc_file}
      DEPENDS ${file} ${h_files})

    list(APPEND source_bc_files ${source_bc_file})
  endforeach()

  string(LENGTH "${mcpu}" gfxlen)
  if(gfxlen EQUAL 6)
    string(SUBSTRING ${mcpu} 3 3 gfxnum)
  else()
    string(SUBSTRING ${mcpu} 3 4 gfxnum)
  endif()
  set(libhostrpc-bc ${CMAKE_BINARY_DIR}/libhostrpc-amdgcn-${mcpu}.bc)
  set(libm-bc ${CMAKE_BINARY_DIR}/libm-amdgcn-${mcpu}.bc)
  get_target_property(ockl_bc_file ockl LOCATION)
  get_target_property(ocml_bc_file ocml LOCATION)
  string(FIND "${amdgpu_32bitwf}" "${mcpu}" is_32bit)
  if(NOT is_32bit EQUAL -1)
    get_target_property(oclc_wf_bc_file oclc_wavefrontsize64_off LOCATION)
  else()
    get_target_property(oclc_wf_bc_file oclc_wavefrontsize64_on LOCATION)
  endif()
  get_target_property(oclc_isa_bc_file oclc_isa_version_${gfxnum} LOCATION)

  # Add custom target so targets from other directories
  # can be added as dependencies to ensure libm
  # and libhostrpc bc files have been built.
  add_custom_target(touch-${mcpu} ALL)
  add_dependencies(touch-${mcpu}
    libm-target-${mcpu}
    libhostrpc-target-${mcpu}
  )

  if (EXISTS ${CMAKE_BINARY_DIR}/../../tools/ROCMDEVLIBS)
    add_dependencies(touch-${mcpu}
      ockl ocml oclc_wavefrontsize64_on oclc_wavefrontsize64_off oclc_isa_version_${gfxnum})
  endif()

  set(linkout_bc_file "linkout.${mcpu}.bc")
  set(opt_bc_file opt-amdgcn-${mcpu}.bc)
  add_custom_command(
    OUTPUT ${linkout_bc_file}
    COMMAND ${LINK_TOOL} ${source_bc_files} ${libm-bc} ${libhostrpc-bc}
      ${ocml_bc_file} ${ockl_bc_file} ${oclc_wf_bc_file} ${oclc_isa_bc_file} -o ${linkout_bc_file}
    DEPENDS ${source_bc_files} touch-${mcpu} ${toolchain_deps})
  add_custom_command(
    OUTPUT ${opt_bc_file}
    COMMAND ${OPT_TOOL} ${link_opt_flags} ${linkout_bc_file} -o ${opt_bc_file}
    DEPENDS ${linkout_bc_file} ${toolchain_deps})
  add_custom_command(
    OUTPUT ${target_libname}
    COMMAND ${PREP_TOOL} ${opt_bc_file} -o ${target_libname}
    DEPENDS ${opt_bc_file} ${toolchain_deps})

  # Add a file-level dependency to ensure that llvm-link and opt are up-to-date.
  # By default, add_custom_command only builds the tool if the executable is missing
  if("${LINK_TOOL}" STREQUAL "$<TARGET_FILE:llvm-link>")
    add_custom_command(OUTPUT ${linkout_bc_file}
      DEPENDS llvm-link
      APPEND)
  endif()
  if("${OPT_TOOL}" STREQUAL "$<TARGET_FILE:opt>")
    add_custom_command(OUTPUT ${opt_bc_file}
      DEPENDS opt
      APPEND)
  endif()
  if("${PREP_TOOL}" STREQUAL "$<TARGET_FILE:prep-libomptarget-bc>")
    add_custom_command(OUTPUT ${target_libname}
      DEPENDS prep-libomptarget-bc
      APPEND)
  endif()

  add_custom_target(lib${libname}-${mcpu} ALL DEPENDS ${target_libname})
endmacro()

set(toolchain_deps "")
if(TARGET llvm-link)
  list(APPEND toolchain_deps llvm-link)
endif()
if(TARGET opt)
  list(APPEND toolchain_deps opt)
endif()
if(TARGET prep-libomptarget-bc)
  list(APPEND toolchain_deps prep-libomptarget-bc)
endif()

set(libname "omptarget-amdgcn")
foreach(mcpu ${amdgpu_mcpus})
  set(bc_libname lib${libname}-${mcpu}.bc)
  build_bc_library(${mcpu} ${bc_libname})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${bc_libname} DESTINATION "${DEVEL_PACKAGE}${OPENMP_INSTALL_LIBDIR}")
endforeach()

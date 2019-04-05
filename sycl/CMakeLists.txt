cmake_minimum_required(VERSION 3.2)

project(sycl-solution)
# Requirements
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

# Get clang's version
include(VersionFromVCS)
set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

# If CLANG_VERSION_* is specified, use it, if not use LLVM_VERSION_*.
if(NOT DEFINED CLANG_VERSION_MAJOR)
  set(CLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif()
if(NOT DEFINED CLANG_VERSION_MINOR)
  set(CLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif()
if(NOT DEFINED CLANG_VERSION_PATCHLEVEL)
  set(CLANG_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
endif()
# Unlike PACKAGE_VERSION, CLANG_VERSION does not include LLVM_VERSION_SUFFIX.
set(CLANG_VERSION "${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}.${CLANG_VERSION_PATCHLEVEL}")

set ( LLVM_INST_INC_DIRECTORY "lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/include" )

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
link_libraries(${OpenCL_LIBRARY})

# Configure SYCL version macro
set(sycl_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/include/CL)
string(TIMESTAMP __SYCL_COMPILER_VERSION "%Y%m%d")
set(version_header "${sycl_inc_dir}/sycl/version.hpp")
configure_file("${version_header}.in" "${version_header}")

# Copy SYCL headers
set(sycl_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/include/CL)
set(dst_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include/CL)
add_custom_target(sycl-headers ALL
COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir} ${dst_dir}
COMMENT "Copying SYCL headers ...")

# Main library

set(sourceRootPath "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(includeRootPath "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(SYCLLibrary sycl)

#To-Do:
#1. Figure out why CMP0057 has to be set. Should have been taken care of earlier in the build
#2. Use AddLLVM to modify the build and access config options
#cmake_policy(SET CMP0057 NEW)
#include(AddLLVM)
set(LLVM_BUILD_LIBRARY_DIRS "${LLVM_BINARY_DIR}/lib/")

set(SYCL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SYCL_TESTS_BINARY_DIR ${SYCL_BINARY_DIR}/test)

set(CLANG_IN_BUILD "${LLVM_BINARY_DIR}/bin/clang")

set(LLVM_TOOLS_DIR "${LLVM_BINARY_DIR}/bin/")

set(SYCL_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include/")
set(OPENCL_INCLUDE "${OpenCL_INCLUDE_DIRS}")

add_library("${SYCLLibrary}" SHARED
  "${includeRootPath}/CL/sycl.hpp"
  "${sourceRootPath}/detail/builtins.cpp"
  "${sourceRootPath}/detail/cnri.cpp"
  "${sourceRootPath}/detail/common.cpp"
  "${sourceRootPath}/detail/context_impl.cpp"
  "${sourceRootPath}/detail/device_info.cpp"
  "${sourceRootPath}/detail/event_impl.cpp"
  "${sourceRootPath}/detail/force_device.cpp"
  "${sourceRootPath}/detail/helpers.cpp"
  "${sourceRootPath}/detail/kernel_impl.cpp"
  "${sourceRootPath}/detail/kernel_info.cpp"
  "${sourceRootPath}/detail/platform_host.cpp"
  "${sourceRootPath}/detail/platform_opencl.cpp"
  "${sourceRootPath}/detail/platform_info.cpp"
  "${sourceRootPath}/detail/program_impl.cpp"
  "${sourceRootPath}/detail/program_manager/program_manager.cpp"
  "${sourceRootPath}/detail/queue_impl.cpp"
  "${sourceRootPath}/detail/os_util.cpp"
  "${sourceRootPath}/detail/scheduler/commands.cpp"
  "${sourceRootPath}/detail/scheduler/printers.cpp"
  "${sourceRootPath}/detail/scheduler/scheduler.cpp"
  "${sourceRootPath}/detail/util.cpp"
  "${sourceRootPath}/context.cpp"
  "${sourceRootPath}/device.cpp"
  "${sourceRootPath}/device_selector.cpp"
  "${sourceRootPath}/event.cpp"
  "${sourceRootPath}/exception.cpp"
  "${sourceRootPath}/half_type.cpp"
  "${sourceRootPath}/kernel.cpp"
  "${sourceRootPath}/platform.cpp"
  "${sourceRootPath}/queue.cpp"
  "${sourceRootPath}/spirv_ops.cpp"
)

include_directories("${SYCLLibrary}" "${includeRootPath}")

set_target_properties("${SYCLLibrary}" PROPERTIES LINKER_LANGUAGE CXX)

# Workaround for bug in GCC version 5.
# More information https://bugs.launchpad.net/ubuntu/+source/gcc-5/+bug/1568899
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 AND
    CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
  target_link_libraries("${SYCLLibrary}" gcc_s gcc)
endif()

install(TARGETS "${SYCLLibrary}" DESTINATION "lib" COMPONENT ${SYCLLibrary})
install(DIRECTORY "${includeRootPath}/." DESTINATION "${LLVM_INST_INC_DIRECTORY}" COMPONENT sycl_headers)

add_subdirectory( test )
add_subdirectory( tools )

set(manifest_list)
set( DEPLOY_LIST
  sycl
  ocl_lib
  ocl_headers
  sycl_headers
  clang
  clang-offload-wrapper
  clang-offload-bundler
  llc
  llvm-as
  llvm-dis
  llvm-spirv
  llvm-link
  opt
)

foreach( comp ${DEPLOY_LIST} )

  message( STATUS "Adding component ${comp} to deploy")

  set (manifest ${CMAKE_CURRENT_BINARY_DIR}/install_manifest_${comp}.txt)
  add_custom_command(OUTPUT ${manifest}
                     COMMAND "${CMAKE_COMMAND}"
                             "-DCMAKE_INSTALL_COMPONENT=${comp}"
                             -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
                     COMMENT "Deploying component ${comp}"
                     USES_TERMINAL)
  list(APPEND manifest_list ${manifest})
endforeach( comp )

add_custom_target(deploy DEPENDS ${manifest_list})

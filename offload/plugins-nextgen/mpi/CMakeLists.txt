##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for a MPI machine if available.
#
##===----------------------------------------------------------------------===##

# Looking for MPI...
find_package(MPI QUIET)


if(NOT(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building MPI offloading plugin: only support MPI in Linux x86_64 or ppc64le hosts.")
  return()
elseif(NOT LIBOMPTARGET_DEP_LIBFFI_FOUND)
  libomptarget_say("Not building MPI offloading plugin: libffi dependency not found.")
  return()
elseif(NOT MPI_CXX_FOUND)
  libomptarget_say("Not building MPI offloading plugin: MPI not found in system.")
  return()
endif()

libomptarget_say("Building MPI NextGen offloading plugin.")

# Create the library and add the default arguments.
add_target_library(omptarget.rtl.mpi MPI)

target_sources(omptarget.rtl.mpi PRIVATE
  src/EventSystem.cpp
  src/rtl.cpp
)

if(FFI_STATIC_LIBRARIES)
  target_link_libraries(omptarget.rtl.mpi PRIVATE FFI::ffi_static)
else()
  target_link_libraries(omptarget.rtl.mpi PRIVATE FFI::ffi)
endif()

target_link_libraries(omptarget.rtl.mpi PRIVATE
  MPI::MPI_CXX
)

# Add include directories
target_include_directories(omptarget.rtl.mpi PRIVATE
                           ${LIBOMPTARGET_INCLUDE_DIR})

# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.mpi
        LIBRARY DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
set_target_properties(omptarget.rtl.mpi PROPERTIES
  INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")


# Set C++20 as the target standard for this plugin.
set_target_properties(omptarget.rtl.mpi
                      PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED ON)

# Configure testing for the MPI plugin.
list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.mpi")
# Report to the parent scope that we are building a plugin for MPI.
set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)

# Define the target specific triples and ELF machine values.
set(LIBOMPTARGET_SYSTEM_TARGETS
    "${LIBOMPTARGET_SYSTEM_TARGETS} x86_64-pc-linux-gnu-mpi" PARENT_SCOPE)

# MPI Device Binary
llvm_add_tool(OPENMP llvm-offload-mpi-device src/EventSystem.cpp src/MPIDeviceMain.cpp)

llvm_update_compile_flags(llvm-offload-mpi-device)

target_link_libraries(llvm-offload-mpi-device PRIVATE
  MPI::MPI_CXX
  LLVMSupport
  omp
)

if(FFI_STATIC_LIBRARIES)
  target_link_libraries(llvm-offload-mpi-device PRIVATE FFI::ffi_static)
else()
  target_link_libraries(llvm-offload-mpi-device PRIVATE FFI::ffi)
endif()

target_include_directories(llvm-offload-mpi-device PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
)


set_target_properties(llvm-offload-mpi-device
  PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

target_compile_definitions(llvm-offload-mpi-device PRIVATE
                           DEBUG_PREFIX="OFFLOAD MPI DEVICE")

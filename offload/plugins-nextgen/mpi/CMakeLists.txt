# Looking for MPI...
find_package(MPI QUIET)

# Check for C++20 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" CXX_SUPPORTS_CXX20_FLAG)

if(NOT(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  message(STATUS "Not building MPI offloading plugin: only support MPI in Linux x86_64 or ppc64le hosts.")
  return()
elseif(NOT MPI_CXX_FOUND)
  message(STATUS "Not building MPI offloading plugin: MPI not found in system.")
  return()
elseif(NOT CXX_SUPPORTS_CXX20_FLAG)
  message(STATUS "Not building MPI offloading plugin: compiler does not support CXX20.")
  return()
endif()

message(STATUS "Building MPI Proxy offloading plugin.")

# Event System
add_subdirectory(event_system)

# MPI Plugin

# Create the library and add the default arguments.
add_target_library(omptarget.rtl.mpi MPI)

target_sources(omptarget.rtl.mpi PRIVATE
  src/rtl.cpp
)

target_link_libraries(omptarget.rtl.mpi PRIVATE
  EventSystem
)

# Add include directories
target_include_directories(omptarget.rtl.mpi PRIVATE
                           ${LIBOMPTARGET_INCLUDE_DIR})

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
    "${LIBOMPTARGET_SYSTEM_TARGETS} x86_64-pc-linux-gnu-mpi nvptx64-nvidia-cuda-mpi" PARENT_SCOPE)

# Remote Plugin Manager
message(STATUS "Building the llvm-offload-mpi-proxy-device")

set(LIBOMPTARGET_ALL_REMOTE_PLUGIN_TARGETS amdgpu cuda host)
set(LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD "all" CACHE STRING
    "Semicolon-separated list of plugins to use: cuda, amdgpu, host or \"all\".")

if(LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD STREQUAL "all")
  set(LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD ${LIBOMPTARGET_ALL_REMOTE_PLUGIN_TARGETS})
endif()

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux" AND
   "host" IN_LIST LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD)
  message(STATUS "Not building remote host plugin: only Linux systems are supported")
  list(REMOVE_ITEM LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD "host")
endif()
if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$"
        AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  if("amdgpu" IN_LIST LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD)
    message(STATUS "Not building remote AMDGPU plugin: only support AMDGPU in "
                   "Linux x86_64, ppc64le, or aarch64 hosts")
    list(REMOVE_ITEM LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD "amdgpu")
  endif()
  if("cuda" IN_LIST LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD)
    message(STATUS "Not building remote CUDA plugin: only support CUDA in "
                   "Linux x86_64, ppc64le, or aarch64 hosts")
    list(REMOVE_ITEM LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD "cuda")
  endif()
endif()
if("mpi" IN_LIST LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD)
  message(STATUS "It is not possible to build the mpi plugin inside "
                 "the remote proxy device")
  list(REMOVE_ITEM LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD "mpi")
endif()

message(STATUS "Building the MPI Plugin with support for remote offloading to "
               "the \"${LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD}\" plugins")

set(REMOTE_MPI_ENUM_PLUGIN_TARGETS "")
foreach(plugin IN LISTS LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD)
  set(REMOTE_MPI_ENUM_PLUGIN_TARGETS
      "${REMOTE_MPI_ENUM_PLUGIN_TARGETS}PLUGIN_TARGET(${plugin})\n")
endforeach()
string(STRIP ${REMOTE_MPI_ENUM_PLUGIN_TARGETS} REMOTE_MPI_ENUM_PLUGIN_TARGETS)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/RemoteTargets.def.in
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}/Shared/RemoteTargets.def
)

llvm_add_tool(OPENMP llvm-offload-mpi-proxy-device
  src/ProxyDevice.cpp
  src/RemotePluginManager.cpp
  ${LIBOMPTARGET_SRC_DIR}/OpenMP/OMPT/Callback.cpp
)

llvm_update_compile_flags(llvm-offload-mpi-proxy-device)

target_link_libraries(llvm-offload-mpi-proxy-device PRIVATE
  EventSystem
  LLVMSupport
  omp
)

add_dependencies(llvm-offload-mpi-proxy-device omp)

target_include_directories(llvm-offload-mpi-proxy-device PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
)

foreach(plugin IN LISTS LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD)
  target_link_libraries(llvm-offload-mpi-proxy-device PRIVATE omptarget.rtl.${plugin})
  add_dependencies(llvm-offload-mpi-proxy-device omptarget.rtl.${plugin})
endforeach()

# Set C++20 as the target standard for this plugin.
set_target_properties(llvm-offload-mpi-proxy-device
                      PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED ON)

target_compile_definitions(llvm-offload-mpi-proxy-device PRIVATE
                           TARGET_NAME=llvm-offload-mpi-proxy-device
                           DEBUG_PREFIX="MPIProxyDevice")

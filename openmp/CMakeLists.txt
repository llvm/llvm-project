cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(OPENMP_LLVM_TOOLS_DIR "" CACHE PATH "Path to LLVM tools for testing")

add_subdirectory(runtime)


set(ENABLE_LIBOMPTARGET ON)
# Currently libomptarget cannot be compiled on Windows or MacOS X.
# Since the device plugins are only supported on Linux anyway,
# there is no point in trying to compile libomptarget on other OSes.
if (WIN32 OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(ENABLE_LIBOMPTARGET OFF)
endif()

option(OPENMP_ENABLE_LIBOMPTARGET "Enable building libomptarget for offloading."
       ${ENABLE_LIBOMPTARGET})
if (OPENMP_ENABLE_LIBOMPTARGET)
  add_subdirectory(libomptarget)
endif()

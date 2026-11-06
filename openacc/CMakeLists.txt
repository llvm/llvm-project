cmake_minimum_required(VERSION 3.20.0)
set(LLVM_SUBPROJECT_TITLE "OpenACC")

set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

list(INSERT CMAKE_MODULE_PATH 0
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  message(FATAL_ERROR "Standalone build is not supported. Use:\n"
    "  cmake <llvm-project>/runtimes -DLLVM_ENABLE_RUNTIMES=openacc")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED NO)
set(CMAKE_CXX_EXTENSIONS NO)

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(OPENACC_TARGET_SUBDIR "${LLVM_DEFAULT_TARGET_TRIPLE}")
  if(OPENACC_LIBDIR_SUBDIR)
    string(APPEND OPENACC_TARGET_SUBDIR "/${OPENACC_LIBDIR_SUBDIR}")
  endif()
  cmake_path(NORMAL_PATH OPENACC_TARGET_SUBDIR)
  set(OPENACC_INSTALL_LIBDIR "lib${LLVM_LIBDIR_SUFFIX}/${OPENACC_TARGET_SUBDIR}" CACHE STRING
      "Path where built OpenACC libraries should be installed.")
else()
  set(OPENACC_INSTALL_LIBDIR "lib${LLVM_LIBDIR_SUFFIX}" CACHE STRING
      "Path where built OpenACC libraries should be installed.")
endif()

# Use the current compiler target to determine the appropriate runtime to build.
if("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^amdgcn|^nvptx|^spirv64" OR
   "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^amdgcn|^nvptx|^spirv64")
  add_subdirectory(device)
endif()

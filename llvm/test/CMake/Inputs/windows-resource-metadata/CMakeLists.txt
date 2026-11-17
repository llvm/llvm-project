cmake_minimum_required(VERSION 3.20)
project(WindowsResourceMetadata NONE)
list(APPEND CMAKE_MODULE_PATH
  "${LLVM_SRC_ROOT}/cmake/modules"
  "${LLVM_SRC_ROOT}/../cmake/Modules")
include(AddLLVM)
set(LLVM_SOURCE_DIR "${LLVM_SRC_ROOT}")
set(MSVC TRUE)
set(CMAKE_HOST_SYSTEM_NAME Windows)

foreach(name IN ITEMS alpha beta)
  add_windows_version_resource_file(sources resource_file ${name})
  add_library(${name} SHARED IMPORTED)
  set_target_properties(${name} PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${name}-custom-d.dll"
    LLVM_WINDOWS_VERSION_RESOURCE_FILE "${resource_file}")
  set_windows_version_resource_properties(${name} "${resource_file}"
    VERSION_STRING "${name}-version" FILE_DESCRIPTION "${name} library")
endforeach()
set(metadata_targets alpha beta)

if(CLANG_SRC_ROOT)
  include("${CLANG_SRC_ROOT}/cmake/modules/AddClang.cmake")
  set(CLANG_VERSION_MAJOR 24)
  set(CLANG_VERSION_MINOR 0)
  set(CLANG_VERSION_PATCHLEVEL 0)
  set(CLANG_VERSION "24.0.0git")
  add_library(libclang SHARED IMPORTED)
  add_windows_version_resource_file(sources resource_file libclang)
  set_target_properties(libclang PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/libclang.dll"
    LLVM_WINDOWS_VERSION_RESOURCE_FILE "${resource_file}")
  set_clang_windows_version_resource_properties(libclang
    FILE_DESCRIPTION "Clang C interface library")

  add_library(static-library STATIC IMPORTED)
  set_target_properties(static-library PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/static-library.lib")
  set_clang_windows_version_resource_properties(static-library)
  list(APPEND metadata_targets libclang)
endif()

foreach(name IN LISTS metadata_targets)
  get_target_property(resource_file ${name} LLVM_WINDOWS_VERSION_RESOURCE_FILE)
  get_source_file_property(definitions "${resource_file}" COMPILE_DEFINITIONS)
  list(FILTER definitions INCLUDE REGEX
    "^RC_(FILE_DESCRIPTION|COPYRIGHT|ORIGINAL_FILENAME|FILE_VERSION)=")
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${name}.txt"
    CONTENT "${definitions}\n")
endforeach()

if(TEST_LIBRARY_RESOURCE_SCOPES)
  enable_language(CXX RC)
  set(LLVM_ENABLE_EH ON)
  set(LLVM_ENABLE_RTTI ON)
  set(LLVM_ENABLE_PIC ON)
  set(LLVM_ENABLE_PLUGINS ON)
  set(LLVM_PLUGIN_EXT ".dll")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp" "int main() { return 0; }\n")
  add_windows_version_resource_file(sources windows_resource_file scope-tool)
  add_executable(scope-tool "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp" ${sources})
  set_windows_version_resource_properties(scope-tool "${windows_resource_file}"
    FILE_DESCRIPTION "Original executable")
  get_source_file_property(before "${windows_resource_file}" COMPILE_DEFINITIONS)

  foreach(type IN ITEMS STATIC MODULE)
    llvm_add_library(scope-${type} ${type}
      "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp" NO_INSTALL_RPATH)
    get_target_property(resource_file scope-${type} LLVM_WINDOWS_VERSION_RESOURCE_FILE)
    get_source_file_property(after "${windows_resource_file}" COMPILE_DEFINITIONS)
    if(resource_file OR NOT "${before}" STREQUAL "${after}")
      message(FATAL_ERROR "${type} library modified the executable resource")
    endif()
    message("${type} resource: none; executable metadata unchanged")
  endforeach()

  llvm_add_library(scope-dual SHARED STATIC
    "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp" NO_INSTALL_RPATH)
  get_target_property(resource_file scope-dual_static LLVM_WINDOWS_VERSION_RESOURCE_FILE)
  get_source_file_property(after "${windows_resource_file}" COMPILE_DEFINITIONS)
  if(resource_file OR NOT "${before}" STREQUAL "${after}")
    message(FATAL_ERROR "dual library modified the executable resource")
  endif()
  message("dual static resource: none; executable metadata unchanged")
endif()

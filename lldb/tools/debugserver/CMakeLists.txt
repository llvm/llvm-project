cmake_minimum_required(VERSION 3.4.3)

project(Debugserver LANGUAGES C CXX ASM-ATT)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/../../cmake"
    "${CMAKE_SOURCE_DIR}/../../cmake/modules"
    )

  include(LLDBStandalone)
  include(AddLLDB)

  set(LLDB_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../../")
  include_directories(${LLDB_SOURCE_DIR}/include)

  option(LLDB_USE_ENTITLEMENTS "When codesigning, use entitlements if available" ON)
  if(LLDB_CODESIGN_IDENTITY)
    # In the future we may use LLVM_CODESIGNING_IDENTITY directly.
    set(LLVM_CODESIGNING_IDENTITY ${LLDB_CODESIGN_IDENTITY})
  endif()
endif()

add_subdirectory(source)

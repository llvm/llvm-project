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

  # lldb-suite is a dummy target that encompasses all the necessary tools and
  # libraries for building a fully-functioning liblldb.
  add_custom_target(lldb-suite)
  set(LLDB_SUITE_TARGET lldb-suite)
endif()

add_subdirectory(source)

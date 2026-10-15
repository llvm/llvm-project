cmake_minimum_required(VERSION 3.31.0)

project(swift-debugserver LANGUAGES C Swift)

# Must go below project(..)
include(GNUInstallDirs)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(LLDB_BUILT_STANDALONE ON)

  set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/../../cmake"
    "${CMAKE_SOURCE_DIR}/../../cmake/modules")

  include(LLDBStandalone)
  include(debugserverConfig)
  include(AddLLDB)

  set(LLDB_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../../")
  include_directories(${LLDB_SOURCE_DIR}/include)
endif()

add_subdirectory(source)

cmake_minimum_required(VERSION 3.31.0)

project(swift-debugserver LANGUAGES C Swift)

include(GNUInstallDirs)

if(PROJECT_IS_TOP_LEVEL)
  set(LLDB_BUILT_STANDALONE ON)

  list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/../../cmake"
    "${CMAKE_SOURCE_DIR}/../../cmake/modules")

  include(LLDBStandalone)
  include(debugserverConfig)
  include(AddLLDB)
endif()

add_compile_options("$<$<COMPILE_LANGUAGE:Swift>:-explicit-module-build>")

add_subdirectory(Sources)

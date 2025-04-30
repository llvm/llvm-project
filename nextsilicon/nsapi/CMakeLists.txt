cmake_minimum_required(VERSION 3.16)

project(nsapi_core)

file(GLOB_RECURSE NSAPI_SOURCE_FILES "*.cpp")

# Add parent directory to have `#include <nsapi/nsapi.hpp> work in 'internal' dir
include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(${PROJECT_NAME} SHARED ${NSAPI_SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

# install library to sysroot lib directory
install(TARGETS ${PROJECT_NAME}
  DESTINATION "lib")

# install header files to root include directory with matching hierarchy
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DESTINATION "include"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
)

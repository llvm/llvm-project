# Check if this is a in tree build.
if (NOT DEFINED LLVM_MAIN_SRC_DIR)
  project(Polly)
  cmake_minimum_required(VERSION 3.13.4)
  if ("${CMAKE_VERSION}" VERSION_LESS "3.20.0")
    message(WARNING
      "Your CMake version is ${CMAKE_VERSION}. Starting with LLVM 17.0.0, the "
      "minimum version of CMake required to build LLVM will become 3.20.0, and "
      "using an older CMake will become an error. Please upgrade your CMake to "
      "at least 3.20.0 now to avoid issues in the future!")
  endif()
  set(POLLY_STANDALONE_BUILD TRUE)
endif()

# Must go below project(..)
include(GNUInstallDirs)

if(POLLY_STANDALONE_BUILD)
  # Where is LLVM installed?
  find_package(LLVM CONFIG REQUIRED)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR})
  include(HandleLLVMOptions)
  include(AddLLVM)

  # Add the llvm header path.
  include_directories(${LLVM_INCLUDE_DIRS})

  # Sources available, too?
  if (LLVM_BUILD_MAIN_SRC_DIR)
    set(default_llvm_src "${LLVM_BUILD_MAIN_SRC_DIR}")
  else()
    set(default_llvm_src "${CMAKE_CURRENT_SOURCE_DIR}/../llvm")
  endif()
  set(LLVM_SOURCE_ROOT ${default_llvm_src} CACHE PATH "Path to LLVM source tree")

  # Enable unit tests if available.
  set(POLLY_GTEST_AVAIL 0)
  set(UNITTEST_DIR ${LLVM_THIRD_PARTY_DIR}/unittest)
  if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
    if (NOT TARGET gtest)
      add_subdirectory(${UNITTEST_DIR} third-party/unittest)
    endif()
    set(POLLY_GTEST_AVAIL 1)
  endif()

  if (LLVM_ENABLE_PIC)
    # Make sure the isl c files are built as fPIC if possible
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif ()

  # Set directory for polly-isl-test.
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
else ()
  set(LLVM_SOURCE_ROOT "${LLVM_MAIN_SRC_DIR}")
  set(POLLY_GTEST_AVAIL 1)
endif ()

set(POLLY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(POLLY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if(NOT DEFINED LLVM_COMMON_CMAKE_UTILS)
  set(LLVM_COMMON_CMAKE_UTILS ${POLLY_SOURCE_DIR}/../cmake)
endif()

# Make sure that our source directory is on the current cmake module path so that
# we can include cmake files from this directory.
list(INSERT CMAKE_MODULE_PATH 0
  "${POLLY_SOURCE_DIR}/cmake"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

include("polly_macros")

# Add appropriate flags for GCC
if (CMAKE_COMPILER_IS_GNUCXX)
  # FIXME: Turn off exceptions, RTTI:
  # -fno-exceptions -fno-rtti
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings -fno-exceptions -fno-rtti")
elseif (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c-")
  add_definitions("-D_HAS_EXCEPTIONS=0")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
endif ()

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(POLLY_BUNDLED_ISL "Use the bundled version of libisl included in Polly" ON)
if (NOT POLLY_BUNDLED_ISL)
  find_package(ISL MODULE REQUIRED)
  message(STATUS "Using external libisl ${ISL_VERSION} in: ${ISL_PREFIX}")
  set(ISL_TARGET ISL)
else()
  set(ISL_INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}/lib/External/isl/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/External/isl/include
  )
  set(ISL_TARGET PollyISL)
endif()

include_directories(
  BEFORE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ISL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/External/pet/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/External
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.h"
    )

  install(DIRECTORY ${POLLY_BINARY_DIR}/include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "CMakeFiles" EXCLUDE
    )
endif()

add_definitions( -D_GNU_SOURCE )

add_subdirectory(docs)
add_subdirectory(lib)
add_subdirectory(test)
if (POLLY_GTEST_AVAIL)
  add_subdirectory(unittests)
endif ()
add_subdirectory(cmake)
# TODO: docs.


configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/include/polly/Config/config.h.cmake
                ${POLLY_BINARY_DIR}/include/polly/Config/config.h )

# Add target to check formatting of polly files
file( GLOB_RECURSE files *.h lib/*.cpp lib/*.c tools/*.cpp tools/*.c tools/*.h unittests/*.cpp)
file( GLOB_RECURSE external lib/External/*.h lib/External/*.c lib/External/*.cpp isl_config.h)
list( REMOVE_ITEM files ${external})

set(check_format_depends)
set(update_format_depends)
set(i 0)
foreach (file IN LISTS files)
  add_custom_command(OUTPUT polly-check-format${i}
    COMMAND clang-format -sort-includes -style=llvm ${file} | diff -u ${file} -
    VERBATIM
    COMMENT "Checking format of ${file}..."
  )
  list(APPEND check_format_depends "polly-check-format${i}")

  add_custom_command(OUTPUT polly-update-format${i}
    COMMAND clang-format -sort-includes -i -style=llvm ${file}
    VERBATIM
    COMMENT "Updating format of ${file}..."
  )
  list(APPEND update_format_depends "polly-update-format${i}")

  math(EXPR i ${i}+1)
endforeach ()

add_custom_target(polly-check-format DEPENDS ${check_format_depends})
set_target_properties(polly-check-format PROPERTIES FOLDER "Polly")

add_custom_target(polly-update-format DEPENDS ${update_format_depends})
set_target_properties(polly-update-format PROPERTIES FOLDER "Polly")


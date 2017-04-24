# Check if this is a in tree build.
if (NOT DEFINED LLVM_MAIN_SRC_DIR)
  project(Polly)
  cmake_minimum_required(VERSION 3.4.3)

  # Where is LLVM installed?
  set(LLVM_INSTALL_ROOT "" CACHE PATH "Root of LLVM install.")
  # Check if the LLVM_INSTALL_ROOT valid.
  if( NOT EXISTS ${LLVM_INSTALL_ROOT}/include/llvm )
    message(FATAL_ERROR "LLVM_INSTALL_ROOT (${LLVM_INSTALL_ROOT}) is not a valid LLVM installation.")
  endif(NOT EXISTS ${LLVM_INSTALL_ROOT}/include/llvm)
  # FileCheck, not and llvm-lit are not install by default, warn the user to copy them.
  if( NOT EXISTS ${LLVM_INSTALL_ROOT}/bin/FileCheck
      OR NOT EXISTS ${LLVM_INSTALL_ROOT}/bin/not
      OR NOT EXISTS ${LLVM_INSTALL_ROOT}/bin/llvm-lit )
    message(WARNING "'FileCheck', 'not' and 'llvm-lit' are required by running regress tests, "
                    "but they are not installed! Please copy them to "
                    "${LLVM_INSTALL_ROOT}/bin.")
  endif()
  # Add the llvm header path.
  include_directories(${LLVM_INSTALL_ROOT}/include/)

  # Get LLVM's own libraries.
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --libs
                  OUTPUT_VARIABLE LLVM_LIBS
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Get the system librarys that will link into LLVM.
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --system-libs
                  OUTPUT_VARIABLE LLVM_SYSTEM_LIBS
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "System libs required by LLVM: ${LLVM_SYSTEM_LIBS}")

  # Determine where LLVM stores its libraries.
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --libdir
                  OUTPUT_VARIABLE LLVM_LIBRARY_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  link_directories("${LLVM_LIBRARY_DIR}")

  # Now set the header paths.
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --includedir
                  OUTPUT_VARIABLE LLVM_INCLUDE_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  include_directories( ${LLVM_INCLUDE_DIR} )

  # Get the TARGET_TRIPLE
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --host-target
                  OUTPUT_VARIABLE TARGET_TRIPLE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  # And then set the cxx flags.
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --cxxflags
                  OUTPUT_VARIABLE LLVM_CXX_FLAGS
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${LLVM_CXX_FLAGS})

  # Check LLVM_ENABLE_ASSERTIONS
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --assertion-mode
                  OUTPUT_VARIABLE LLVM_ENABLE_ASSERTIONS
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  # Copied from LLVM's HandleLLVMOptions.cmake
  if( LLVM_ENABLE_ASSERTIONS )
    # MSVC doesn't like _DEBUG on release builds. See PR 4379.
    if( NOT MSVC )
      add_definitions( -D_DEBUG )
    endif()
    # On non-Debug builds cmake automatically defines NDEBUG, so we
    # explicitly undefine it:
    if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
      add_definitions( -UNDEBUG )
      # Also remove /D NDEBUG to avoid MSVC warnings about conflicting defines.
      foreach (flags_var_to_scrub
          CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_RELWITHDEBINFO
          CMAKE_CXX_FLAGS_MINSIZEREL
          CMAKE_C_FLAGS_RELEASE
          CMAKE_C_FLAGS_RELWITHDEBINFO
          CMAKE_C_FLAGS_MINSIZEREL)
        string (REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " "
          "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
      endforeach()
    endif()
  endif()

  # Sources available, too?
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --src-root
                  OUTPUT_VARIABLE MAIN_SRC_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(LLVM_SOURCE_ROOT ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")

  # Enable unit tests if available.
  set(UNITTEST_DIR ${LLVM_SOURCE_ROOT}/utils/unittest)
  if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
    add_library(gtest
      ${UNITTEST_DIR}/googletest/src/gtest-all.cc
      ${UNITTEST_DIR}/googlemock/src/gmock-all.cc
    )
    target_include_directories(gtest
      PUBLIC
      "${UNITTEST_DIR}/googletest/include"
      "${UNITTEST_DIR}/googlemock/include"

      PRIVATE
      "${UNITTEST_DIR}/googletest"
      "${UNITTEST_DIR}/googlemock"
    )
    target_link_libraries(gtest ${LLVM_SYSTEM_LIBS})

    add_library(gtest_main ${UNITTEST_DIR}/UnitTestMain/TestMain.cpp)
    target_link_libraries(gtest_main gtest)

    set(POLLY_GTEST_AVAIL 1)
  endif()

  # Make sure the isl c files are built as fPIC
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

  # Set directory for polly-isl-test.
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
else ()
  set(LLVM_SOURCE_ROOT "${LLVM_MAIN_SRC_DIR}")
  set(POLLY_GTEST_AVAIL 1)
endif ()

set(POLLY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(POLLY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
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

# Add path for custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${POLLY_SOURCE_DIR}/cmake")

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(POLLY_ENABLE_GPGPU_CODEGEN "Enable GPGPU code generation feature" OFF)
if (POLLY_ENABLE_GPGPU_CODEGEN)
  # Do not require CUDA, as GPU code generation test cases can be run without
  # a cuda library.
  FIND_PACKAGE(CUDA)
  set(GPU_CODEGEN TRUE)
else(POLLY_ENABLE_GPGPU_CODEGEN)
  set(GPU_CODEGEN FALSE)
endif(POLLY_ENABLE_GPGPU_CODEGEN)


# Support GPGPU code generation if the library is available.
if (CUDALIB_FOUND)
  INCLUDE_DIRECTORIES( ${CUDALIB_INCLUDE_DIR} )
endif(CUDALIB_FOUND)

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
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/External/JSON/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/External/pet/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/External
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
    )

  install(DIRECTORY ${POLLY_BINARY_DIR}/include/
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN ".svn" EXCLUDE
    )
endif()

add_definitions( -D_GNU_SOURCE )

add_subdirectory(docs)
add_subdirectory(lib)
add_subdirectory(test)
if (POLLY_GTEST_AVAIL)
  add_subdirectory(unittests)
endif ()
add_subdirectory(tools)
add_subdirectory(cmake)
# TODO: docs.


configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/include/polly/Config/config.h.cmake
                ${POLLY_BINARY_DIR}/include/polly/Config/config.h )

# Add target to check formatting of polly files
file( GLOB_RECURSE files *.h lib/*.cpp lib/*.c tools/*.cpp tools/*.c tools/*.h unittests/*.cpp)
file( GLOB_RECURSE external lib/External/*.h lib/External/*.c lib/External/*.cpp)
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


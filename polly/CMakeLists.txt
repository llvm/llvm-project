# Check if this is a in tree build.
if (NOT DEFINED LLVM_MAIN_SRC_DIR)
  project(Polly)
  cmake_minimum_required(VERSION 3.13.4)

  # Where is LLVM installed?
  find_package(LLVM CONFIG REQUIRED)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR})
  include(HandleLLVMOptions)
  include(AddLLVM)

  # Add the llvm header path.
  include_directories(${LLVM_INCLUDE_DIRS})

  # Sources available, too?
  if (LLVM_BUILD_MAIN_SRC_DIR)
    set(LLVM_SOURCE_ROOT ${LLVM_BUILD_MAIN_SRC_DIR} CACHE PATH
      "Path to LLVM source tree")
  else()
    execute_process(COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-config" --src-root
      OUTPUT_VARIABLE MAIN_SRC_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(LLVM_SOURCE_ROOT ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")
  endif()

  # Enable unit tests if available.
  set(POLLY_GTEST_AVAIL 0)
  set(UNITTEST_DIR ${LLVM_SOURCE_ROOT}/utils/unittest)
  if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
    # The build tree already exports the gtest target, which we can reuse
    if (TARGET gtest)
      # LLVM Doesn't export gtest's include directorys, so do that here
      set_target_properties(gtest
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
        "${UNITTEST_DIR}/googletest/include;${UNITTEST_DIR}/googlemock/include"
        )
      set(POLLY_GTEST_AVAIL 1)
    else()
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
      target_link_libraries(gtest PUBLIC -lpthread)

      add_library(gtest_main ${UNITTEST_DIR}/UnitTestMain/TestMain.cpp)
      target_link_libraries(gtest_main PUBLIC gtest)

      set(POLLY_GTEST_AVAIL 1)
    endif()
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
set(GPU_CODEGEN FALSE)
if (POLLY_ENABLE_GPGPU_CODEGEN)
  # Do not require CUDA/OpenCL, as GPU code generation test cases can be run
  # without a CUDA/OpenCL library.
  if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
    FIND_PACKAGE(CUDA)
    FIND_PACKAGE(OpenCL)
    set(GPU_CODEGEN TRUE)
  else()
    message(WARNING "The LLVM NVPTX target is required for GPU code generation")
  endif()
endif(POLLY_ENABLE_GPGPU_CODEGEN)


# Support GPGPU code generation if the library is available.
if (CUDA_FOUND)
  add_definitions(-DHAS_LIBCUDART)
  INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS} )
endif(CUDA_FOUND)
if (OpenCL_FOUND)
  add_definitions(-DHAS_LIBOPENCL)
  INCLUDE_DIRECTORIES( ${OpenCL_INCLUDE_DIR} )
endif(OpenCL_FOUND)

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


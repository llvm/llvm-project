# Keep track if we have all dependencies.
set(ORC_RT_LLVM_TOOLS_AVAILABLE TRUE)

if (NOT DEFINED ORC_RT_LLVM_TOOLS_DIR AND DEFINED LLVM_BINARY_DIR)
  cmake_path(APPEND ORC_RT_LLVM_TOOLS_DIR "${LLVM_BINARY_DIR}" "bin")
endif()

if (TARGET utils/llvm-lit/all)
   list(APPEND ORC_RT_TEST_DEPS utils/llvm-lit/all)
endif()

# Add dependence on FileCheck.
if (TARGET FileCheck)
  list(APPEND ORC_RT_TEST_DEPS FileCheck)
endif()

find_program(ORC_RT_FILECHECK_EXECUTABLE
  NAMES FileCheck
  PATHS ${ORC_RT_LLVM_TOOLS_DIR})
if (NOT ORC_RT_FILECHECK_EXECUTABLE)
  message(STATUS "Cannot find FileCheck. Please put it in your PATH, set "
                 "ORC_RT_FILECHECK_EXECUTABLE to its full path, or point "
                 "ORC_RT_LLVM_TOOLS_DIR to its directory.")
  set(ORC_RT_LLVM_TOOLS_AVAILABLE FALSE)
endif()

# Add dependence on not.
if (TARGET not)
  list(APPEND ORC_RT_TEST_DEPS not)
endif()

find_program(ORC_RT_NOT_EXECUTABLE
  NAMES not
  PATHS ${ORC_RT_LLVM_TOOLS_DIR})
if (NOT ORC_RT_NOT_EXECUTABLE)
  message(STATUS "Cannot find 'not'. Please put it in your PATH, set "
                 "ORC_RT_NOT_EXECUTABLE to its full path, or point "
                 "ORC_RT_LLVM_TOOLS_DIR to its directory.")
  set(ORC_RT_LLVM_TOOLS_AVAILABLE FALSE)
endif()

# Add dependence on split-file.
if (TARGET split-file)
  list(APPEND ORC_RT_TEST_DEPS split-file)
endif()

find_program(ORC_RT_SPLIT_FILE_EXECUTABLE
  NAMES split-file
  PATHS ${ORC_RT_LLVM_TOOLS_DIR})
if (NOT ORC_RT_SPLIT_FILE_EXECUTABLE)
  message(STATUS "Cannot find split-file. Please put it in your PATH, set "
                 "ORC_RT_SPLIT_FILE_EXECUTABLE to its full path, or point "
                 "ORC_RT_LLVM_TOOLS_DIR to its directory.")
  set(ORC_RT_LLVM_TOOLS_AVAILABLE FALSE)
endif()

# Add dependencies on optional tools, if they're being built alongside us.
# Tests that need these tools are gated on lit features, so they're not required.
foreach(tool clang llvm-jitlink llvm-mc)
  if (TARGET ${tool})
    list(APPEND ORC_RT_TEST_DEPS ${tool})
  endif()
endforeach()

# The sysroot that the regression tests' default compilers build against: the
# one the runtime is built against, so that tests see the same system headers.
# A clang built from source doesn't find the SDK on its own on Darwin.
if (APPLE)
  set(ORC_RT_TEST_SYSROOT "${CMAKE_OSX_SYSROOT}")
  # CMake treats a CMAKE_OSX_SYSROOT containing "/" as a path, and resolves
  # anything else to one itself, except under the Xcode generator. It's empty
  # if not set (since CMake 4.0 the compiler picks its default SDK). Ask xcrun
  # for the path in the remaining cases.
  if (NOT ORC_RT_TEST_SYSROOT MATCHES "/")
    set(orc_rt_xcrun_sdk_args)
    if (ORC_RT_TEST_SYSROOT)
      set(orc_rt_xcrun_sdk_args --sdk "${ORC_RT_TEST_SYSROOT}")
    endif()
    execute_process(COMMAND xcrun ${orc_rt_xcrun_sdk_args} --show-sdk-path
      OUTPUT_VARIABLE ORC_RT_TEST_SYSROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE orc_rt_xcrun_result)
    if (NOT orc_rt_xcrun_result EQUAL 0 OR
        NOT IS_DIRECTORY "${ORC_RT_TEST_SYSROOT}")
      message(WARNING "orc-rt: could not find an SDK for the regression tests; "
                      "tests that include system headers will fail. Set "
                      "CMAKE_OSX_SYSROOT to choose one.")
      set(ORC_RT_TEST_SYSROOT "")
    endif()
  endif()
else()
  set(ORC_RT_TEST_SYSROOT "${CMAKE_SYSROOT}")
endif()

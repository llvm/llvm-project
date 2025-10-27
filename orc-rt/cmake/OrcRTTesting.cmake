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
  message(STATUS "Cannot find FileCheck. Please put it in your PATH, set ORC_RT_FILECHECK_EXECUTABLE to its full path, or point ORC_RT_LLVM_TOOLS_DIR to its directory.")
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
  message(STATUS "Cannot find 'not'. Please put it in your PATH, set ORC_RT_NOT_EXECUTABLE to its full path, or point ORC_RT_LLVM_TOOLS_DIR to its directory.")
  set(ORC_RT_LLVM_TOOLS_AVAILABLE FALSE)
endif()

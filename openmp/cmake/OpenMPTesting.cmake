# Keep track if we have all dependencies.
set(ENABLE_CHECK_TARGETS TRUE)

# Function to find required dependencies for testing.
function(find_standalone_test_dependencies)
  include(FindPythonInterp)

  if (NOT PYTHONINTERP_FOUND)
    message(STATUS "Could not find Python.")
    message(WARNING "The check targets will not be available!")
    set(ENABLE_CHECK_TARGETS FALSE PARENT_SCOPE)
    return()
  endif()

  # Find executables.
  find_program(OPENMP_LLVM_LIT_EXECUTABLE
    NAMES llvm-lit lit.py lit
    PATHS ${OPENMP_LLVM_TOOLS_DIR})
  if (NOT OPENMP_LLVM_LIT_EXECUTABLE)
    message(STATUS "Cannot find llvm-lit.")
    message(STATUS "Please put llvm-lit in your PATH, set OPENMP_LLVM_LIT_EXECUTABLE to its full path, or point OPENMP_LLVM_TOOLS_DIR to its directory.")
    message(WARNING "The check targets will not be available!")
    set(ENABLE_CHECK_TARGETS FALSE PARENT_SCOPE)
    return()
  endif()

  find_program(OPENMP_FILECHECK_EXECUTABLE
    NAMES FileCheck
    PATHS ${OPENMP_LLVM_TOOLS_DIR})
  if (NOT OPENMP_FILECHECK_EXECUTABLE)
    message(STATUS "Cannot find FileCheck.")
    message(STATUS "Please put FileCheck in your PATH, set OPENMP_FILECHECK_EXECUTABLE to its full path, or point OPENMP_LLVM_TOOLS_DIR to its directory.")
    message(WARNING "The check targets will not be available!")
    set(ENABLE_CHECK_TARGETS FALSE PARENT_SCOPE)
    return()
  endif()
endfunction()

if (${OPENMP_STANDALONE_BUILD})
  find_standalone_test_dependencies()

  # Make sure we can use the console pool for recent CMake and Ninja > 1.5.
  if (CMAKE_VERSION VERSION_LESS 3.1.20141117)
    set(cmake_3_2_USES_TERMINAL)
  else()
    set(cmake_3_2_USES_TERMINAL USES_TERMINAL)
  endif()

  # Set lit arguments.
  set(DEFAULT_LIT_ARGS "-sv --show-unsupported --show-xfail")
  if (MSVC OR XCODE)
    set(DEFAULT_LIT_ARGS "${DEFAULT_LIT_ARGS} --no-progress-bar")
  endif()
  # TODO: Remove once bots are updated to use the new option.
  if (DEFINED LIBOMP_LIT_ARGS)
    set(DEFAULT_LIT_ARGS ${LIBOMP_LIT_ARGS})
  endif()
  set(OPENMP_LIT_ARGS "${DEFAULT_LIT_ARGS}" CACHE STRING "Options for lit.")
  separate_arguments(OPENMP_LIT_ARGS)
else()
  set(OPENMP_FILECHECK_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/FileCheck)
endif()

# Macro to extract information about compiler from file. (no own scope)
macro(extract_test_compiler_information lang file)
  file(READ ${file} information)
  list(GET information 0 path)
  list(GET information 1 id)
  list(GET information 2 version)
  list(GET information 3 openmp_flags)

  set(OPENMP_TEST_${lang}_COMPILER_PATH ${path})
  set(OPENMP_TEST_${lang}_COMPILER_ID ${id})
  set(OPENMP_TEST_${lang}_COMPILER_VERSION ${version})
  set(OPENMP_TEST_${lang}_COMPILER_OPENMP_FLAGS ${openmp_flags})
endmacro()

# Function to set variables with information about the test compiler.
function(set_test_compiler_information dir)
  extract_test_compiler_information(C ${dir}/CCompilerInformation.txt)
  extract_test_compiler_information(CXX ${dir}/CXXCompilerInformation.txt)
  if (NOT("${OPENMP_TEST_C_COMPILER_ID}" STREQUAL "${OPENMP_TEST_CXX_COMPILER_ID}" AND
          "${OPENMP_TEST_C_COMPILER_VERSION}" STREQUAL "${OPENMP_TEST_CXX_COMPILER_VERSION}"))
    message(STATUS "Test compilers for C and C++ don't match.")
    message(WARNING "The check targets will not be available!")
    set(ENABLE_CHECK_TARGETS FALSE PARENT_SCOPE)
  else()
    set(OPENMP_TEST_COMPILER_ID "${OPENMP_TEST_C_COMPILER_ID}" PARENT_SCOPE)
    set(OPENMP_TEST_COMPILER_VERSION "${OPENMP_TEST_C_COMPILER_VERSION}" PARENT_SCOPE)
    set(OPENMP_TEST_COMPILER_OPENMP_FLAGS "${OPENMP_TEST_C_COMPILER_OPENMP_FLAGS}" PARENT_SCOPE)

    # Determine major version.
    string(REGEX MATCH "[0-9]+" major "${OPENMP_TEST_C_COMPILER_VERSION}")
    set(OPENMP_TEST_COMPILER_VERSION_MAJOR "${major}" PARENT_SCOPE)
  endif()
endfunction()

if (${OPENMP_STANDALONE_BUILD})
  # Detect compiler that should be used for testing.
  # We cannot use ExternalProject_Add() because its configuration runs when this
  # project is built which is too late for detecting the compiler...
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/DetectTestCompiler)
  execute_process(
    COMMAND ${CMAKE_COMMAND} ${CMAKE_CURRENT_LIST_DIR}/DetectTestCompiler
      -DCMAKE_C_COMPILER=${OPENMP_TEST_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${OPENMP_TEST_CXX_COMPILER}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/DetectTestCompiler
    OUTPUT_VARIABLE DETECT_COMPILER_OUT
    ERROR_VARIABLE DETECT_COMPILER_ERR
    RESULT_VARIABLE DETECT_COMPILER_RESULT)
  if (DETECT_COMPILER_RESULT)
    message(STATUS "Could not detect test compilers.")
    message(WARNING "The check targets will not be available!")
    set(ENABLE_CHECK_TARGETS FALSE)
  else()
    set_test_compiler_information(${CMAKE_CURRENT_BINARY_DIR}/DetectTestCompiler)
  endif()
else()
  # Set the information that we know.
  set(OPENMP_TEST_COMPILER_ID "Clang")
  # Cannot use CLANG_VERSION because we are not guaranteed that this is already set.
  set(OPENMP_TEST_COMPILER_VERSION "${LLVM_VERSION}")
  set(OPENMP_TEST_COMPILER_VERSION_MAJOR "${LLVM_MAJOR_VERSION}")
  set(OPENMP_TEST_COMPILER_OPENMP_FLAGS "-fopenmp")
endif()

# Function to set compiler features for use in lit.
function(set_test_compiler_features)
  if ("${OPENMP_TEST_COMPILER_ID}" STREQUAL "GNU")
    set(comp "gcc")
  else()
    # Just use the lowercase of the compiler ID as fallback.
    string(TOLOWER "${OPENMP_TEST_COMPILER_ID}" comp)
  endif()
  set(OPENMP_TEST_COMPILER_FEATURES "['${comp}', '${comp}-${OPENMP_TEST_COMPILER_VERSION_MAJOR}', '${comp}-${OPENMP_TEST_COMPILER_VERSION}']" PARENT_SCOPE)
endfunction()
set_test_compiler_features()

# Function to add a testsuite for an OpenMP runtime library.
function(add_openmp_testsuite target comment)
  if (NOT ENABLE_CHECK_TARGETS)
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E echo "${target} does nothing, dependencies not found.")
    message(STATUS "${target} does nothing.")
    return()
  endif()

  cmake_parse_arguments(ARG "" "" "DEPENDS" ${ARGN})
  # EXCLUDE_FROM_ALL excludes the test ${target} out of check-openmp.
  if (NOT EXCLUDE_FROM_ALL)
    # Register the testsuites and depends for the check-openmp rule.
    set_property(GLOBAL APPEND PROPERTY OPENMP_LIT_TESTSUITES ${ARG_UNPARSED_ARGUMENTS})
    set_property(GLOBAL APPEND PROPERTY OPENMP_LIT_DEPENDS ${ARG_DEPENDS})
  endif()

  if (${OPENMP_STANDALONE_BUILD})
    add_custom_target(${target}
      COMMAND ${PYTHON_EXECUTABLE} ${OPENMP_LLVM_LIT_EXECUTABLE} ${OPENMP_LIT_ARGS} ${ARG_UNPARSED_ARGUMENTS}
      COMMENT ${comment}
      DEPENDS ${ARG_DEPENDS}
      ${cmake_3_2_USES_TERMINAL}
    )
  else()
    add_lit_testsuite(${target}
      ${comment}
      ${ARG_UNPARSED_ARGUMENTS}
      DEPENDS clang clang-headers FileCheck ${ARG_DEPENDS}
    )
  endif()
endfunction()

function(construct_check_openmp_target)
  get_property(OPENMP_LIT_TESTSUITES GLOBAL PROPERTY OPENMP_LIT_TESTSUITES)
  get_property(OPENMP_LIT_DEPENDS GLOBAL PROPERTY OPENMP_LIT_DEPENDS)

  # We already added the testsuites themselves, no need to do that again.
  set(EXCLUDE_FROM_ALL True)
  add_openmp_testsuite(check-openmp "Running OpenMP tests" ${OPENMP_LIT_TESTSUITES} DEPENDS ${OPENMP_LIT_DEPENDS})
endfunction()

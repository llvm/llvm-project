# Keep track if we have all dependencies.
set(ENABLE_CHECK_TARGETS TRUE)

# Function to find required dependencies for testing.
function(find_standalone_test_dependencies)
  find_package (Python3 COMPONENTS Interpreter)

  if (NOT Python3_Interpreter_FOUND)
    message(STATUS "Could not find Python.")
    message(WARNING "The check targets will not be available!")
    set(ENABLE_CHECK_TARGETS FALSE PARENT_SCOPE)
    return()
  else()
    set(Python3_EXECUTABLE ${Python3_EXECUTABLE} PARENT_SCOPE)
  endif()

  # Find executables.
  find_program(OPENMP_LLVM_LIT_EXECUTABLE
    NAMES llvm-lit.py llvm-lit lit.py lit
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

  find_program(OPENMP_NOT_EXECUTABLE
    NAMES not
    PATHS ${OPENMP_LLVM_TOOLS_DIR})
  if (NOT OPENMP_NOT_EXECUTABLE)
    message(STATUS "Cannot find 'not'.")
    message(STATUS "Please put 'not' in your PATH, set OPENMP_NOT_EXECUTABLE to its full path, or point OPENMP_LLVM_TOOLS_DIR to its directory.")
    message(WARNING "The check targets will not be available!")
    set(ENABLE_CHECK_TARGETS FALSE PARENT_SCOPE)
    return()
  endif()
endfunction()

if (${OPENMP_STANDALONE_BUILD})
  find_standalone_test_dependencies()

  # Set lit arguments.
  set(DEFAULT_LIT_ARGS "-sv --show-unsupported --show-xfail")
  if (MSVC OR XCODE)
    set(DEFAULT_LIT_ARGS "${DEFAULT_LIT_ARGS} --no-progress-bar")
  endif()
  if ("${CMAKE_SYSTEM_NAME}" MATCHES "AIX")
    set(DEFAULT_LIT_ARGS "${DEFAULT_LIT_ARGS} --time-tests --timeout=1800")
  endif()
  set(OPENMP_LIT_ARGS "${DEFAULT_LIT_ARGS}" CACHE STRING "Options for lit.")
  separate_arguments(OPENMP_LIT_ARGS)
else()
  if (NOT TARGET "FileCheck")
    message(STATUS "Cannot find 'FileCheck'.")
    message(WARNING "The check targets will not be available!")
    set(ENABLE_CHECK_TARGETS FALSE)
  else()
    set(OPENMP_FILECHECK_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/FileCheck)
  endif()
  set(OPENMP_NOT_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/not)
endif()
set(OFFLOAD_DEVICE_INFO_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-offload-device-info)
set(OFFLOAD_TBLGEN_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/offload-tblgen)

# Macro to extract information about compiler from file. (no own scope)
macro(extract_test_compiler_information lang file)
  file(READ ${file} information)
  list(GET information 0 path)
  list(GET information 1 id)
  list(GET information 2 version)
  list(GET information 3 openmp_flags)
  list(GET information 4 has_tsan_flags)
  list(GET information 5 has_omit_frame_pointer_flags)
  list(GET information 6 has_omp_h)

  set(OPENMP_TEST_${lang}_COMPILER_PATH ${path})
  set(OPENMP_TEST_${lang}_COMPILER_ID ${id})
  set(OPENMP_TEST_${lang}_COMPILER_VERSION ${version})
  set(OPENMP_TEST_${lang}_COMPILER_OPENMP_FLAGS ${openmp_flags})
  set(OPENMP_TEST_${lang}_COMPILER_HAS_TSAN_FLAGS ${has_tsan_flags})
  set(OPENMP_TEST_${lang}_COMPILER_HAS_OMIT_FRAME_POINTER_FLAGS ${has_omit_frame_pointer_flags})
  set(OPENMP_TEST_${lang}_COMPILER_HAS_OMP_H ${has_omp_h})
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
    set(OPENMP_TEST_COMPILER_HAS_TSAN_FLAGS "${OPENMP_TEST_C_COMPILER_HAS_TSAN_FLAGS}" PARENT_SCOPE)
    set(OPENMP_TEST_COMPILER_HAS_OMIT_FRAME_POINTER_FLAGS "${OPENMP_TEST_C_COMPILER_HAS_OMIT_FRAME_POINTER_FLAGS}" PARENT_SCOPE)
    set(OPENMP_TEST_COMPILER_HAS_OMP_H "${OPENMP_TEST_C_COMPILER_HAS_OMP_H}" PARENT_SCOPE)

    # Determine major version.
    string(REGEX MATCH "[0-9]+" major "${OPENMP_TEST_C_COMPILER_VERSION}")
    string(REGEX MATCH "[0-9]+\\.[0-9]+" majorminor "${OPENMP_TEST_C_COMPILER_VERSION}")
    set(OPENMP_TEST_COMPILER_VERSION_MAJOR "${major}" PARENT_SCOPE)
    set(OPENMP_TEST_COMPILER_VERSION_MAJOR_MINOR "${majorminor}" PARENT_SCOPE)
  endif()
endfunction()

if (${OPENMP_STANDALONE_BUILD})
  # Detect compiler that should be used for testing.
  # We cannot use ExternalProject_Add() because its configuration runs when this
  # project is built which is too late for detecting the compiler...
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/DetectTestCompiler)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -G${CMAKE_GENERATOR} ${CMAKE_CURRENT_SOURCE_DIR}/../openmp/cmake/DetectTestCompiler
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
  set(OPENMP_TEST_COMPILER_VERSION_MAJOR "${LLVM_VERSION_MAJOR}")
  set(OPENMP_TEST_COMPILER_VERSION_MAJOR_MINOR "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")
  # Unfortunately the top-level cmake/config-ix.cmake file mangles CMake's
  # CMAKE_THREAD_LIBS_INIT variable from the FindThreads package, so work
  # around that, until it is fixed there.
  if("${CMAKE_THREAD_LIBS_INIT}" STREQUAL "-lpthread")
    set(OPENMP_TEST_COMPILER_THREAD_FLAGS "-pthread")
  else()
    set(OPENMP_TEST_COMPILER_THREAD_FLAGS "${CMAKE_THREAD_LIBS_INIT}")
  endif()
  if(TARGET tsan)
    set(OPENMP_TEST_COMPILER_HAS_TSAN_FLAGS 1)
  else()
    set(OPENMP_TEST_COMPILER_HAS_TSAN_FLAGS 0)
  endif()
  set(OPENMP_TEST_COMPILER_HAS_OMP_H 1)
  set(OPENMP_TEST_COMPILER_OPENMP_FLAGS "-fopenmp ${OPENMP_TEST_COMPILER_THREAD_FLAGS}")
  set(OPENMP_TEST_COMPILER_HAS_OMIT_FRAME_POINTER_FLAGS 1)
endif()

# Function to set compiler features for use in lit.
function(update_test_compiler_features)
  set(FEATURES "[")
  set(first TRUE)
  foreach(feat IN LISTS OPENMP_TEST_COMPILER_FEATURE_LIST)
    if (NOT first)
      string(APPEND FEATURES ", ")
    endif()
    set(first FALSE)
    string(APPEND FEATURES "'${feat}'")
  endforeach()
  string(APPEND FEATURES "]")
  set(OPENMP_TEST_COMPILER_FEATURES ${FEATURES} PARENT_SCOPE)
endfunction()

function(set_test_compiler_features)
  if ("${OPENMP_TEST_COMPILER_ID}" STREQUAL "GNU")
    set(comp "gcc")
  elseif ("${OPENMP_TEST_COMPILER_ID}" STREQUAL "Intel")
    set(comp "icc")
  else()
    # Just use the lowercase of the compiler ID as fallback.
    string(TOLOWER "${OPENMP_TEST_COMPILER_ID}" comp)
  endif()
  set(OPENMP_TEST_COMPILER_FEATURE_LIST ${comp} ${comp}-${OPENMP_TEST_COMPILER_VERSION_MAJOR} ${comp}-${OPENMP_TEST_COMPILER_VERSION_MAJOR_MINOR} ${comp}-${OPENMP_TEST_COMPILER_VERSION} PARENT_SCOPE)
endfunction()
set_test_compiler_features()
update_test_compiler_features()

# Function to add a testsuite for an OpenMP runtime library.
function(add_offload_testsuite target comment)
  if (NOT ENABLE_CHECK_TARGETS)
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E echo "${target} does nothing, dependencies not found.")
    message(STATUS "${target} does nothing.")
    return()
  endif()

  cmake_parse_arguments(ARG "EXCLUDE_FROM_CHECK_ALL" "" "DEPENDS;ARGS" ${ARGN})
  # EXCLUDE_FROM_CHECK_ALL excludes the test ${target} out of check-offload.
  if (NOT ARG_EXCLUDE_FROM_CHECK_ALL)
    # Register the testsuites and depends for the check-offload rule.
    set_property(GLOBAL APPEND PROPERTY OPENMP_LIT_TESTSUITES ${ARG_UNPARSED_ARGUMENTS})
    set_property(GLOBAL APPEND PROPERTY OPENMP_LIT_DEPENDS ${ARG_DEPENDS})
  endif()

  if (${OPENMP_STANDALONE_BUILD})
    set(LIT_ARGS ${OPENMP_LIT_ARGS} ${ARG_ARGS})
    add_custom_target(${target}
      COMMAND ${Python3_EXECUTABLE} ${OPENMP_LLVM_LIT_EXECUTABLE} ${LIT_ARGS} ${ARG_UNPARSED_ARGUMENTS}
      COMMENT ${comment}
      DEPENDS ${ARG_DEPENDS}
      USES_TERMINAL
    )
  else()
    if (ARG_EXCLUDE_FROM_CHECK_ALL)
      add_lit_testsuite(${target}
        ${comment}
        ${ARG_UNPARSED_ARGUMENTS}
        EXCLUDE_FROM_CHECK_ALL
        DEPENDS clang FileCheck not ${ARG_DEPENDS}
        ARGS ${ARG_ARGS}
      )
    else()
      add_lit_testsuite(${target}
        ${comment}
        ${ARG_UNPARSED_ARGUMENTS}
        DEPENDS clang FileCheck not ${ARG_DEPENDS}
        ARGS ${ARG_ARGS}
      )
    endif()
  endif()
endfunction()

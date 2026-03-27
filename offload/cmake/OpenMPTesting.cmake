# Keep track if we have all dependencies.
set(ENABLE_CHECK_TARGETS TRUE)

if (TARGET FileCheck)
  set(OFFLOAD_FILECHECK_EXECUTABLE ${LLVM_TOOLS_BINARY_DIR}/FileCheck)
else()
  message(STATUS "Cannot find 'FileCheck'.")
  message(WARNING "The check targets will not be available!")
  set(ENABLE_CHECK_TARGETS FALSE)
endif()

set(OFFLOAD_NOT_EXECUTABLE ${LLVM_TOOLS_BINARY_DIR}/not)
set(OFFLOAD_DEVICE_INFO_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-offload-device-info)
set(OFFLOAD_TBLGEN_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/offload-tblgen)

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
    set_property(GLOBAL APPEND PROPERTY OFFLOAD_LIT_TESTSUITES ${ARG_UNPARSED_ARGUMENTS})
    set_property(GLOBAL APPEND PROPERTY OFFLOAD_LIT_DEPENDS ${ARG_DEPENDS})
  endif()

  set(extra_args)
  if(ARG_EXCLUDE_FROM_CHECK_ALL)
    list(APPEND extra_args EXCLUDE_FROM_CHECK_ALL)
  endif()

  add_lit_testsuite(${target}
    ${comment}
    ${ARG_UNPARSED_ARGUMENTS}
    ${extra_args}
    DEPENDS clang FileCheck not ${ARG_DEPENDS}
    ARGS ${ARG_ARGS}
  )
endfunction()

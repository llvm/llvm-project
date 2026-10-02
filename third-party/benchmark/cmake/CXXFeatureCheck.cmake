# - Compile and run code to check for C++ features
#
# This functions compiles a source file under the `cmake` folder
# and adds the corresponding `HAVE_[FILENAME]` flag to the CMake
# environment
#
#  cxx_feature_check(<FLAG> [<VARIANT>])
#
# - Example
#
# include(CXXFeatureCheck)
# cxx_feature_check(STD_REGEX)
# Requires CMake 3.13+

if(__cxx_feature_check)
  return()
endif()
set(__cxx_feature_check INCLUDED)

option(CXXFEATURECHECK_DEBUG OFF "Enable debug messages for CXX feature checks")

function(cxx_feature_check_print log)
  if(CXXFEATURECHECK_DEBUG)
    message(STATUS "${log}")
  endif()
endfunction()

function(cxx_feature_check FEATURE)
  string(TOLOWER ${FEATURE} FILE)
  string(TOUPPER HAVE_${FEATURE} VAR)

  # Check if the variable is already defined to a true or false for a quick return.
  # This allows users to predefine the variable to skip the check.
  # Or, if the variable is already defined by a previous check, we skip the costly check.
  if (DEFINED ${VAR})
    if (${VAR})
      cxx_feature_check_print("Feature ${FEATURE} already enabled.")
      add_compile_definitions(${VAR})
    else()
      cxx_feature_check_print("Feature ${FEATURE} already disabled.")
    endif()
    return()
  endif()

  set(FEATURE_CHECK_CMAKE_FLAGS ${BENCHMARK_CXX_LINKER_FLAGS})
  if (ARGC GREATER 1)
    message(STATUS "Enabling additional flags: ${ARGV1}")
    list(APPEND FEATURE_CHECK_CMAKE_FLAGS ${ARGV1})
  endif()

  if(CMAKE_CROSSCOMPILING)
    cxx_feature_check_print("Cross-compiling to test ${FEATURE}")
    try_compile(
      COMPILE_STATUS
      ${CMAKE_BINARY_DIR} 
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${FILE}.cpp
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CMAKE_FLAGS "${FEATURE_CHECK_CMAKE_FLAGS}"
      LINK_LIBRARIES "${BENCHMARK_CXX_LIBRARIES}"
      OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
    )
    if(COMPILE_STATUS)
      set(RUN_STATUS 0)
      message(WARNING
              "If you see build failures due to cross compilation, try setting ${VAR} to 0")
    endif()
  else()
    cxx_feature_check_print("Compiling and running to test ${FEATURE}")
    try_run(
      RUN_STATUS 
      COMPILE_STATUS
      ${CMAKE_BINARY_DIR} 
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${FILE}.cpp
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CMAKE_FLAGS "${FEATURE_CHECK_CMAKE_FLAGS}"
      LINK_LIBRARIES "${BENCHMARK_CXX_LIBRARIES}"
      COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
      RUN_OUTPUT_VARIABLE RUN_OUTPUT
    )
  endif()

  if(COMPILE_STATUS AND RUN_STATUS EQUAL 0)
    message(STATUS "Performing Test ${FEATURE} -- success")
    set(${VAR} TRUE CACHE BOOL "" FORCE)
    add_compile_definitions(${VAR})
    return()
  endif()

  set(${VAR} FALSE CACHE BOOL "" FORCE)
  message(STATUS "Performing Test ${FEATURE} -- failed")

  if(NOT COMPILE_STATUS)
    cxx_feature_check_print("Compile Output: ${COMPILE_OUTPUT}")
  else()
    cxx_feature_check_print("Run Output: ${RUN_OUTPUT}")
  endif()

endfunction()

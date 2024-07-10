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
# Requires CMake 2.8.12+

if(__cxx_feature_check)
  return()
endif()
set(__cxx_feature_check INCLUDED)

option(CXXFEATURECHECK_DEBUG OFF)

function(cxx_feature_check FILE)
  string(TOLOWER ${FILE} FILE)
  string(TOUPPER ${FILE} VAR)
  string(TOUPPER "HAVE_${VAR}" FEATURE)
  if (DEFINED HAVE_${VAR})
    set(HAVE_${VAR} 1 PARENT_SCOPE)
    add_definitions(-DHAVE_${VAR})
    return()
  endif()

  set(FEATURE_CHECK_CMAKE_FLAGS ${BENCHMARK_CXX_LINKER_FLAGS})
  if (ARGC GREATER 1)
    message(STATUS "Enabling additional flags: ${ARGV1}")
    list(APPEND FEATURE_CHECK_CMAKE_FLAGS ${ARGV1})
  endif()

  if (NOT DEFINED COMPILE_${FEATURE})
    if(CMAKE_CROSSCOMPILING)
      message(STATUS "Cross-compiling to test ${FEATURE}")
      try_compile(COMPILE_${FEATURE}
              ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${FILE}.cpp
              CXX_STANDARD 11
              CXX_STANDARD_REQUIRED ON
              CMAKE_FLAGS ${FEATURE_CHECK_CMAKE_FLAGS}
              LINK_LIBRARIES ${BENCHMARK_CXX_LIBRARIES}
              OUTPUT_VARIABLE COMPILE_OUTPUT_VAR)
      if(COMPILE_${FEATURE})
        message(WARNING
              "If you see build failures due to cross compilation, try setting HAVE_${VAR} to 0")
        set(RUN_${FEATURE} 0 CACHE INTERNAL "")
      else()
        set(RUN_${FEATURE} 1 CACHE INTERNAL "")
      endif()
    else()
      message(STATUS "Compiling and running to test ${FEATURE}")
      try_run(RUN_${FEATURE} COMPILE_${FEATURE}
              ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${FILE}.cpp
              CXX_STANDARD 11
              CXX_STANDARD_REQUIRED ON
              CMAKE_FLAGS ${FEATURE_CHECK_CMAKE_FLAGS}
              LINK_LIBRARIES ${BENCHMARK_CXX_LIBRARIES}
              COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR)
    endif()
  endif()

  if(RUN_${FEATURE} EQUAL 0)
    message(STATUS "Performing Test ${FEATURE} -- success")
    set(HAVE_${VAR} 1 PARENT_SCOPE)
    add_definitions(-DHAVE_${VAR})
  else()
    if(NOT COMPILE_${FEATURE})
      if(CXXFEATURECHECK_DEBUG)
        message(STATUS "Performing Test ${FEATURE} -- failed to compile: ${COMPILE_OUTPUT_VAR}")
      else()
        message(STATUS "Performing Test ${FEATURE} -- failed to compile")
      endif()
    else()
      message(STATUS "Performing Test ${FEATURE} -- compiled but failed to run")
    endif()
  endif()
endfunction()

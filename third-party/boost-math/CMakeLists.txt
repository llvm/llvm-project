# Copyright 2020 Peter Dimov
# Copyright 2021 Matt Borland
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

cmake_minimum_required(VERSION 3.8...3.16)

project(boost_math VERSION 1.92.0 LANGUAGES CXX)

add_library(boost_math INTERFACE)

add_library(Boost::math ALIAS boost_math)

target_include_directories(boost_math INTERFACE include)
if(NOT CMAKE_VERSION VERSION_LESS "3.19")
  file(GLOB_RECURSE headers include/*.hpp)
  target_sources(boost_math PRIVATE ${headers})
endif()

include(CMakeDependentOption)

cmake_dependent_option(BOOST_MATH_STANDALONE "Use Boost.Math in standalone mode" ON "NOT BOOST_SUPERPROJECT_VERSION" OFF)

message(STATUS "Boost.Math: standalone mode ${BOOST_MATH_STANDALONE}")

if(BOOST_MATH_STANDALONE)

  target_compile_definitions(boost_math INTERFACE BOOST_MATH_STANDALONE=1)

else()

  target_link_libraries(boost_math
    INTERFACE
      Boost::assert
      Boost::concept_check
      Boost::config
      Boost::core
      Boost::integer
      Boost::lexical_cast
      Boost::predef
      Boost::random
      Boost::throw_exception
  )

endif()

target_compile_features(boost_math INTERFACE cxx_std_14)

# Legacy C99/TR1 compiled libraries
option(BOOST_MATH_BUILD_WITH_LEGACY_FUNCTIONS "Build the C99 and TR1 compiled libraries" OFF)

if(BOOST_MATH_BUILD_WITH_LEGACY_FUNCTIONS)

include(CheckCXXSourceCompiles)
set(_config_inc "")
if(NOT BOOST_MATH_STANDALONE)
  get_target_property(_config_type Boost::config TYPE)
  if(_config_type STREQUAL "INTERFACE_LIBRARY")
    get_target_property(_config_inc Boost::config INTERFACE_INCLUDE_DIRECTORIES)
  endif()
endif()
set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include" ${_config_inc})
check_cxx_source_compiles("#include <${CMAKE_CURRENT_SOURCE_DIR}/config/has_long_double_support.cpp> \n int main() { return 0;}" BOOST_MATH_HAS_LONG_DOUBLE)
unset(CMAKE_REQUIRED_INCLUDES)

set(C99_SOURCES
  acosh
  asinh
  atanh
  cbrt
  copysign
  erfc
  erf
  expm1
  fmax
  fmin
  fpclassify
  hypot
  lgamma
  llround
  log1p
  lround
  nextafter
  nexttoward
  round
  tgamma
  trunc
)

set(TR1_SOURCES
  assoc_laguerre
  assoc_legendre
  beta
  comp_ellint_1
  comp_ellint_2
  comp_ellint_3
  cyl_bessel_i
  cyl_bessel_j
  cyl_bessel_k
  cyl_neumann
  ellint_1
  ellint_2
  ellint_3
  expint
  hermite
  laguerre
  legendre
  riemann_zeta
  sph_bessel
  sph_legendre
  sph_neumann
)

list(TRANSFORM C99_SOURCES PREPEND "src/tr1/")
list(TRANSFORM TR1_SOURCES PREPEND "src/tr1/")

list(TRANSFORM C99_SOURCES APPEND "f.cpp" OUTPUT_VARIABLE C99_SOURCESf)
list(TRANSFORM TR1_SOURCES APPEND "f.cpp" OUTPUT_VARIABLE TR1_SOURCESf)

set(types "" f)

if(BOOST_MATH_HAS_LONG_DOUBLE)
  list(TRANSFORM C99_SOURCES APPEND "l.cpp" OUTPUT_VARIABLE C99_SOURCESl)
  list(TRANSFORM TR1_SOURCES APPEND "l.cpp" OUTPUT_VARIABLE TR1_SOURCESl)
  list(APPEND types l)
endif()

list(TRANSFORM C99_SOURCES APPEND ".cpp")
list(TRANSFORM TR1_SOURCES APPEND ".cpp")

foreach(type IN LISTS types)
  add_library(boost_math_tr1${type} ${TR1_SOURCES${type}})
  add_library(Boost::math_tr1${type} ALIAS boost_math_tr1${type})
  if(NOT BOOST_MATH_STANDALONE)
    target_link_libraries(boost_math_tr1${type} PUBLIC Boost::config)
  endif()
  target_include_directories(boost_math_tr1${type} PRIVATE src/tr1)
  target_include_directories(boost_math_tr1${type} PRIVATE include)

  add_library(boost_math_c99${type} ${C99_SOURCES${type}})
  add_library(Boost::math_c99${type} ALIAS boost_math_c99${type})
  if(NOT BOOST_MATH_STANDALONE)
    target_link_libraries(boost_math_c99${type} PUBLIC Boost::config)
  endif()
  target_include_directories(boost_math_c99${type} PRIVATE src/tr1)
  target_include_directories(boost_math_c99${type} PRIVATE include)

  if(BUILD_SHARED_LIBS)
    target_compile_definitions(boost_math_tr1${type} PUBLIC BOOST_MATH_TR1_DYN_LINK=1)
    target_compile_definitions(boost_math_c99${type} PUBLIC BOOST_MATH_TR1_DYN_LINK=1)
    if(WIN32)
      target_compile_definitions(boost_math_tr1${type} PRIVATE "BOOST_SYMBOL_EXPORT=__declspec(dllexport)" BOOST_ALL_NO_LIB)
      target_compile_definitions(boost_math_c99${type} PRIVATE "BOOST_SYMBOL_EXPORT=__declspec(dllexport)" BOOST_ALL_NO_LIB)
    else()
      target_compile_definitions(boost_math_tr1${type} PRIVATE "BOOST_SYMBOL_EXPORT=__attribute__((visibility(\"default\")))" BOOST_ALL_NO_LIB)
      target_compile_definitions(boost_math_c99${type} PRIVATE "BOOST_SYMBOL_EXPORT=__attribute__((visibility(\"default\")))" BOOST_ALL_NO_LIB)
    endif()
  endif()

  target_compile_features(boost_math_tr1${type} PUBLIC cxx_std_14)
  target_compile_features(boost_math_c99${type} PUBLIC cxx_std_14)

  if(APPLE)
    target_compile_definitions(boost_math_tr1${type} PRIVATE _DARWIN_C_SOURCE)
    target_compile_definitions(boost_math_c99${type} PRIVATE _DARWIN_C_SOURCE)
  endif()
endforeach()

if(BOOST_SUPERPROJECT_VERSION AND NOT CMAKE_VERSION VERSION_LESS 3.13)
  set(boost_math_legacy_targets "")
  foreach(type IN LISTS types)
    list(APPEND boost_math_legacy_targets boost_math_c99${type} boost_math_tr1${type})
  endforeach()
  boost_install(TARGETS ${boost_math_legacy_targets} VERSION "${BOOST_SUPERPROJECT_VERSION}")
endif()

endif()

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")

  add_subdirectory(test)

# Only enable tests when we're the root project
elseif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

  include(CTest)
  add_subdirectory(test)
  
  include(GNUInstallDirs)
  install(DIRECTORY "include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

endif()

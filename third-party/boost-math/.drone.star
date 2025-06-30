# Use, modification, and distribution are
# subject to the Boost Software License, Version 1.0. (See accompanying
# file LICENSE.txt)
#
# Copyright Rene Rivera 2020.
# Copyright John Maddock 2021.

# For Drone CI we use the Starlark scripting language to reduce duplication.
# As the yaml syntax for Drone CI is rather limited.
#
#
globalenv={}
linuxglobalimage="cppalliance/droneubuntu1604:1"
windowsglobalimage="cppalliance/dronevs2019"

def main(ctx):

  things_to_test = [ "special_fun", "distribution_tests", "mp", "misc", "interpolators", "quadrature", "autodiff", "long-running-tests", "float128_tests", "concepts" ]
  gcc13_things_to_test = [ "special_fun", "distribution_tests", "mp", "misc", "interpolators", "quadrature", "autodiff", "long-running-tests", "float128_tests", "concepts", "new_floats" ]
  sanitizer_test = [ "special_fun", "distribution_tests", "misc", "interpolators", "quadrature", "float128_tests" ]
  gnu_5_stds = [ "gnu++14", "c++14" ]
  gnu_6_stds = [ "gnu++14", "c++14", "gnu++17", "c++17" ]
  clang_6_stds = [ "c++14", "c++17" ]
  gnu_9_stds = [ "gnu++14", "c++14", "gnu++17", "c++17", "gnu++2a", "c++2a" ]
  clang_10_stds = [ "c++14", "c++17", "c++2a" ]
  gnu_non_native = [ "gnu++17" ]
  gcc13_stds = [ "c++23" ]

  result = []

  for suite in sanitizer_test:
    #
    # Sanitizers:
    #
    result.append(linux_cxx("Ubuntu Clang-18 C++20 ASAN" + " " + suite, "clang++-18", packages="clang-18", privileged=True, buildtype="boost", image="cppalliance/droneubuntu2404:1", environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-18', 'CXXSTD': 'gnu++20', 'TEST_SUITE': suite, 'OPTIONS': '<cxxflags>-fsanitize=address <linkflags>-fsanitize=address <cxxflags>-DBOOST_CI_SANITIZER_BUILD' }, globalenv=globalenv))
    result.append(linux_cxx("Ubuntu Clang-18 C++20 USAN" + " " + suite, "clang++-18", packages="clang-18", privileged=True, buildtype="boost", image="cppalliance/droneubuntu2404:1", environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-18', 'CXXSTD': 'gnu++20', 'TEST_SUITE': suite, 'OPTIONS': '<cxxflags>-fsanitize=undefined <linkflags>-fsanitize=undefined <cxxflags>-DBOOST_CI_SANITIZER_BUILD' }, globalenv=globalenv))
    result.append(linux_cxx("Ubuntu Clang-18 C++20 TSAN" + " " + suite, "clang++-18", packages="clang-18", privileged=True, buildtype="boost", image="cppalliance/droneubuntu2404:1", environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-18', 'CXXSTD': 'gnu++20', 'TEST_SUITE': suite, 'OPTIONS': '<cxxflags>-fsanitize=thread <linkflags>-fsanitize=thread <cxxflags>-DBOOST_CI_SANITIZER_BUILD' }, globalenv=globalenv))
    result.append(linux_cxx("Ubuntu Clang-18 C++20 ISAN" + " " + suite, "clang++-18", packages="clang-18", privileged=True, buildtype="boost", image="cppalliance/droneubuntu2404:1", environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-18', 'CXXSTD': 'gnu++20', 'TEST_SUITE': suite, 'OPTIONS': '<cxxflags>-fsanitize=integer <linkflags>-fsanitize=integer' }, globalenv=globalenv))

  for suite in things_to_test:
    for cxx in gnu_5_stds:
      result.append(linux_cxx("Ubuntu g++-5 " + cxx + " " + suite, "g++-5", packages="g++-5", buildtype="boost", image="cppalliance/droneubuntu1804:1", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-5', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
    for cxx in gnu_6_stds:
      result.append(linux_cxx("Ubuntu g++-6 " + cxx + " " + suite, "g++-6", packages="g++-6", buildtype="boost", image="cppalliance/droneubuntu1804:1", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-6', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
      result.append(linux_cxx("Ubuntu g++-7 " + cxx + " " + suite, "g++-7", packages="g++-7", buildtype="boost", image="cppalliance/droneubuntu1804:1", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-7', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
      result.append(linux_cxx("Ubuntu g++-8 " + cxx + " " + suite, "g++-8", packages="g++-8", buildtype="boost", image="cppalliance/droneubuntu2004:1", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-8', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
      result.append(linux_cxx("Ubuntu g++-9 " + cxx + " " + suite, "g++-9", packages="g++-9", buildtype="boost", image="cppalliance/droneubuntu2004:1", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-9', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
    for cxx in clang_6_stds:
      result.append(linux_cxx("Ubuntu clang++-6 " + cxx + " " + suite, "clang++-6.0", packages="clang-6.0", llvm_os="xenial", llvm_ver="6.0", buildtype="boost", image="cppalliance/droneubuntu1804:1", environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-6.0', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
      result.append(linux_cxx("Ubuntu clang++-7 " + cxx + " " + suite, "clang++-7", packages="clang-7", llvm_os="xenial", llvm_ver="7", buildtype="boost", image="cppalliance/droneubuntu1804:1", environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-7', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
      result.append(linux_cxx("Ubuntu clang++-8 " + cxx + " " + suite, "clang++-8", packages="clang-8", llvm_os="xenial", llvm_ver="8", buildtype="boost", image="cppalliance/droneubuntu1804:1", environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-8', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
      result.append(linux_cxx("Ubuntu clang++-9 " + cxx + " " + suite, "clang++-9", packages="clang-9", llvm_os="xenial", llvm_ver="9", buildtype="boost", image="cppalliance/droneubuntu1804:1", environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-9', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
    for cxx in gnu_9_stds:
      result.append(linux_cxx("Ubuntu g++-10 " + cxx + " " + suite, "g++-10", packages="g++-10", buildtype="boost", image="cppalliance/droneubuntu2004:1", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-10', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
      result.append(linux_cxx("Ubuntu g++-11 " + cxx + " " + suite, "g++-11", packages="g++-11", buildtype="boost", image="cppalliance/droneubuntu2004:1", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-11', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
    for cxx in clang_10_stds:
      result.append(linux_cxx("Ubuntu clang++-10 " + cxx + " " + suite, "clang++-10", packages="clang-10", llvm_os="xenial", llvm_ver="10", buildtype="boost", image="cppalliance/droneubuntu1804:1", environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-10', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
    for cxx in gnu_non_native:
      result.append(linux_cxx("Ubuntu g++ s390s " + cxx + " " + suite, "g++", packages="g++", buildtype="boost", image="cppalliance/droneubuntu2204:multiarch", arch="s390x", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
    for cxx in gnu_non_native:
      result.append(linux_cxx("Ubuntu g++ ARM64" + cxx + " " + suite, "g++", packages="g++", buildtype="boost", image="cppalliance/droneubuntu2204:multiarch", arch="arm64", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
    for cxx in gnu_non_native:
      result.append(osx_cxx("M1 Clang " + cxx + " " + suite, "clang++", buildscript="drone", buildtype="boost", xcode_version="14.1", environment={'TOOLSET': 'clang', 'CXXSTD': cxx, 'TEST_SUITE': suite, 'DEFINE': 'BOOST_MATH_NO_REAL_CONCEPT_TESTS,BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS,BOOST_MATH_MULTI_ARCH_CI_RUN', }, globalenv=globalenv))
  for suite in gcc13_things_to_test:
    for cxx in gcc13_stds:
      result.append(linux_cxx("Ubuntu g++-13 " + cxx + " " + suite, "g++-13", packages="g++-13", buildtype="boost", image="cppalliance/droneubuntu2404:1", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-13', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))
      result.append(linux_cxx("Ubuntu g++-14 " + cxx + " " + suite, "g++-14", packages="g++-14", buildtype="boost", image="cppalliance/droneubuntu2404:1", environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-14', 'CXXSTD': cxx, 'TEST_SUITE': suite, }, globalenv=globalenv))

  return result

# from https://github.com/boostorg/boost-ci
load("@boost_ci//ci/drone/:functions.star", "linux_cxx","windows_cxx","osx_cxx","freebsd_cxx")

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
add_lldb_library(lldbSymbolHelpers
  YAMLModuleTester.cpp
  )

# Our current version of gtest does not properly recognize C++11 support
# with MSVC, so it falls back to tr1 / experimental classes.  Since LLVM
# itself requires C++11, we can safely force it on unconditionally so that
# we don't have to fight with the buggy gtest check.
add_definitions(-DGTEST_LANG_CXX11=1)
add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)
include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googlemock/include)

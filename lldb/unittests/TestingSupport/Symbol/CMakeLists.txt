set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
add_lldb_library(lldbSymbolHelpers
  YAMLModuleTester.cpp
  )

target_include_directories(lldbSymbolHelpers PUBLIC
  ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include
  ${LLVM_MAIN_SRC_DIR}/utils/unittest/googlemock/include)

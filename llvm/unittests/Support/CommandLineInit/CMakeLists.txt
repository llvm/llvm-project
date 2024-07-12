set(test_name CommandLineInitTests)
set(test_suite UnitTests)

# We don't call add_llvm_unittest() here, because the function automatically
# links the test against TestMain.cpp, in which main() function calls
# llvm::cl::ParseCommandLineOptions, and it makes the test always pass.
# The following code mainly comes from `add_unittest` in
# llvm/cmake/modules/AddLLVM.cmake, except that gtest_main is excluded from
# target_link_libraries to prevent the test linking against TestMain.cpp.

if (NOT LLVM_BUILD_TESTS)
  set(EXCLUDE_FROM_ALL ON)
endif()

if (SUPPORTS_VARIADIC_MACROS_FLAG)
  list(APPEND LLVM_COMPILE_FLAGS "-Wno-variadic-macros")
endif ()
# Some parts of gtest rely on this GNU extension, don't warn on it.
if(SUPPORTS_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS_FLAG)
  list(APPEND LLVM_COMPILE_FLAGS "-Wno-gnu-zero-variadic-macro-arguments")
endif()

list(APPEND LLVM_LINK_COMPONENTS Support)

add_llvm_executable(${test_name}
  IGNORE_EXTERNALIZE_DEBUGINFO NO_INSTALL_RPATH
  CommandLineInitTest.cpp)

target_link_libraries(${test_name} PRIVATE llvm_gtest)

add_dependencies(${test_suite} ${test_name})

set(outdir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
set_output_directory(${test_name} BINARY_DIR ${outdir} LIBRARY_DIR ${outdir})


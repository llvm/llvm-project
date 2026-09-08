set(test_name LLVMToolSessionTests)
set(test_suite UnitTests)

# This test supplies its own main() so a single LLVMToolSession can own
# InitLLVM for the complete test process.
if (NOT LLVM_BUILD_TESTS)
  set(EXCLUDE_FROM_ALL ON)
endif()

list(APPEND LLVM_LINK_COMPONENTS Support)

add_llvm_executable(${test_name}
  IGNORE_EXTERNALIZE_DEBUGINFO NO_INSTALL_RPATH
  LLVMToolSessionTest.cpp)

target_link_libraries(${test_name} PRIVATE llvm_gtest)
add_dependencies(${test_suite} ${test_name})

set(outdir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
set_output_directory(${test_name} BINARY_DIR ${outdir} LIBRARY_DIR ${outdir})

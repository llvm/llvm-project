# Do not build unittest libraries automatically, they will be pulled in
# by unittests if these are built.
if (NOT ${LLVM_INSTALL_GTEST})
  set (BUILDTREE_ONLY BUILDTREE_ONLY)
  set(EXCLUDE_FROM_ALL ON)
endif()

# This is to avoid the error in gtest-death-test-internal.h
# (150,16): error: 'Create' overrides a member function but
# is not marked 'override' [-Werror,-Wsuggest-override]
# during self-compile on Windows.

if (HOST_WINNT AND "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
  llvm_add_warning_options(CXX -Wno-suggest-override)
endif()

add_llvm_library(LLVMTestingSupport
  Error.cpp
  SupportHelpers.cpp

  ${BUILDTREE_ONLY}

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Testing/Support

  LINK_COMPONENTS
  Support
  )

target_link_libraries(LLVMTestingSupport PRIVATE llvm_gtest)

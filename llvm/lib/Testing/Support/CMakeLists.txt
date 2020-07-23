add_definitions(-DGTEST_LANG_CXX11=1)
add_definitions(-DGTEST_HAS_TR1_TUPLE=0)

add_llvm_library(LLVMTestingSupport
  Annotations.cpp
  Error.cpp
  SupportHelpers.cpp

  BUILDTREE_ONLY
  
  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Testing/Support

  LINK_COMPONENTS
  Support
  )

include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)
include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googlemock/include)
target_link_libraries(LLVMTestingSupport PRIVATE gtest)

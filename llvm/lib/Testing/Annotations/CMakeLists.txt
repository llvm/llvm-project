# Do not build unittest libraries automatically, they will be pulled in
# by unittests if these are built.
set(EXCLUDE_FROM_ALL ON)

add_llvm_library(LLVMTestingAnnotations
  Annotations.cpp

  BUILDTREE_ONLY

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Testing/Support

  LINK_COMPONENTS
  Support
  )

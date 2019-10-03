set(LLVM_LINK_COMPONENTS
  TableGen
  Support
  )

add_llvm_unittest(TableGenTests
  CodeExpanderTest.cpp
  )
include_directories(${CMAKE_SOURCE_DIR}/utils/TableGen)
target_link_libraries(TableGenTests PRIVATE LLVMTableGenGlobalISel)

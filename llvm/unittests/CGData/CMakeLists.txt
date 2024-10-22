set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  CGData
  CodeGen
  Core
  Support
  )

add_llvm_unittest(CGDataTests
  OutlinedHashTreeRecordTest.cpp
  OutlinedHashTreeTest.cpp
  )

target_link_libraries(CGDataTests PRIVATE LLVMTestingSupport)

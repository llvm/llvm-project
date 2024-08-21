set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  CGData
  CodeGen
  Core
  Support
  )

add_llvm_unittest(CodeGenDataTests
  OutlinedHashTreeRecordTest.cpp
  OutlinedHashTreeTest.cpp
  )

target_link_libraries(CodeGenDataTests PRIVATE LLVMTestingSupport)

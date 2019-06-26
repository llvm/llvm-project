set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  DebugInfoGSYM
  MC
  Object
  ObjectYAML
  Support
  )

add_llvm_unittest(DebugInfoGSYMTests
  GSYMTest.cpp
  )

target_link_libraries(DebugInfoGSYMTests PRIVATE LLVMTestingSupport)

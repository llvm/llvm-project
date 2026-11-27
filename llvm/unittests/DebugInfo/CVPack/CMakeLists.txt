set(LLVM_LINK_COMPONENTS
  BinaryFormat
  DebugInfoCodeView
  DebugInfoCVPack
  Support
  )

add_llvm_unittest(DebugInfoCVPackTests
  CVPackTest.cpp
  )

target_link_libraries(DebugInfoCVPackTests PRIVATE LLVMTestingSupport)

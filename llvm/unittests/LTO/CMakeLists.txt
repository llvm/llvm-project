set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  BitReader
  BitWriter
  Core
  LTO
  Support
  TargetParser
  TransformUtils
  nativecodegen
)

add_llvm_unittest(LTOTests
  VirtualFileSystemTest.cpp
)

target_link_libraries(LTOTests PRIVATE LLVMTestingSupport)

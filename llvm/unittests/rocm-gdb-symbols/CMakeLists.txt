set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  Passes
  TransformUtils
  ScalarOpts
  )

add_llvm_unittest(rocm-gdb-symbols
  AsmParserTest.cpp
  AsmWriterTest.cpp
  MetadataTest.cpp
  )

target_link_libraries(rocm-gdb-symbols PRIVATE LLVMTestingSupport)

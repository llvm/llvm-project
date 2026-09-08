set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  JITLink
  MC
  OrcJIT
  OrcShared
  Support
  Target
  nativecodegen
  )

add_llvm_example(OrcV2CBindingsAddObjectFile
  OrcV2CBindingsAddObjectFile.c
  )

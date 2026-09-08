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

add_llvm_example(OrcV2CBindingsBasicUsage
  OrcV2CBindingsBasicUsage.c
  )

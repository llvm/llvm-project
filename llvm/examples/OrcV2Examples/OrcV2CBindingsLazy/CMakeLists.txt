set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  JITLink
  MC
  OrcJIT
  Support
  Target
  nativecodegen
  )

add_llvm_example(OrcV2CBindingsLazy
  OrcV2CBindingsLazy.c
  )

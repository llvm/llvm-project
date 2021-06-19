set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  InstCombine
  IRReader
  JITLink
  MC
  OrcJIT
  Support
  Target
  nativecodegen
  )

add_llvm_example(OrcV2CBindingsIRTransforms
  OrcV2CBindingsIRTransforms.c
  )

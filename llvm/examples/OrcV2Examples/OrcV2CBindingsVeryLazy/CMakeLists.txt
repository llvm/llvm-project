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

add_llvm_example(OrcV2CBindingsVeryLazy
  OrcV2CBindingsVeryLazy.c

  DISABLE_PCH_REUSE # no C++ files, prevent CMake error.
  )

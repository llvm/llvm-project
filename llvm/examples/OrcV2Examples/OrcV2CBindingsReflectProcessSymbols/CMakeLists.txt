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

add_llvm_example(OrcV2CBindingsReflectProcessSymbols
  OrcV2CBindingsReflectProcessSymbols.c
  )

export_executable_symbols(OrcV2CBindingsReflectProcessSymbols)

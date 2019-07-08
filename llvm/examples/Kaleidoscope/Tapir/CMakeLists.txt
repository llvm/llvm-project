set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  Object
  OrcJIT
  RuntimeDyld
  ScalarOpts
  Support
  TapirOpts
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Tapir
  toy.cpp
  )

export_executable_symbols(Kaleidoscope-Tapir)

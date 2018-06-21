set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  Object
  RuntimeDyld
  ScalarOpts
  Support
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch4
  toy.cpp
  )

export_executable_symbols(Kaleidoscope-Ch4)

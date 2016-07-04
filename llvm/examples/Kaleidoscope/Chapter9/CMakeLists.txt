set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Object
  Support
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch9
  toy.cpp
  )

export_executable_symbols(Kaleidoscope-Ch9)

set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Object
  OrcJIT
  OrcShared
  Support
  TargetParser
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch9
  toy.cpp
  )

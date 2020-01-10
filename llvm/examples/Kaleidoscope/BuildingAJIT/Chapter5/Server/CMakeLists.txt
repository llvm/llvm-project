set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  Object
  OrcError
  OrcJIT
  RuntimeDyld
  ScalarOpts
  Support
  TransformUtils
  native
  )

add_kaleidoscope_chapter(BuildingAJIT-Ch5-Server
  server.cpp
  )

export_executable_symbols(BuildingAJIT-Ch5-Server)

set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  MC
  MCJIT
  Passes
  RuntimeDyld
  ScalarOpts
  Support
  Target
  TargetParser
  nativecodegen
  )

set(MCJITTestsSources
  MCJITTest.cpp
  MCJITCAPITest.cpp
  MCJITMemoryManagerTest.cpp
  MCJITMultipleModuleTest.cpp
  MCJITObjectCacheTest.cpp
  )

if(MSVC)
  list(APPEND MCJITTestsSources MCJITTests.def)
endif()

add_llvm_unittest(MCJITTests
  ${MCJITTestsSources}
  )

if(MINGW OR CYGWIN)
  set_property(TARGET MCJITTests PROPERTY LINK_FLAGS -Wl,--export-all-symbols)
endif()

set_property(TARGET MCJITTests PROPERTY FOLDER "Tests/UnitTests/ExecutionTests")

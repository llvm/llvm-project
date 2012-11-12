set(LLVM_LINK_COMPONENTS
  asmparser
  bitreader
  bitwriter
  mcjit
  jit
  nativecodegen
  )

set(MCJITTestsSources
  MCJITTest.cpp
  SectionMemoryManager.cpp
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

set(LLVM_LINK_COMPONENTS
  asmparser
  bitreader
  bitwriter
  jit
  nativecodegen
  )

# HACK: Declare a couple of source files as optionally compiled to satisfy the
# missing-file-checker in LLVM's weird CMake build.
set(LLVM_OPTIONAL_SOURCES
  IntelJITEventListenerTest.cpp
  OProfileJITEventListenerTest.cpp
  )

if( LLVM_USE_INTEL_JITEVENTS )
  set(ProfileTestSources
    IntelJITEventListenerTest.cpp
    )
  set(LLVM_LINK_COMPONENTS
    ${LLVM_LINK_COMPONENTS}
    IntelJITEvents
    ) 
endif( LLVM_USE_INTEL_JITEVENTS )

if( LLVM_USE_OPROFILE )
  set(ProfileTestSources
    ${ProfileTestSources}
    OProfileJITEventListenerTest.cpp
    )
  set(LLVM_LINK_COMPONENTS
    ${LLVM_LINK_COMPONENTS}
    OProfileJIT
    )
endif( LLVM_USE_OPROFILE )

set(JITTestsSources
  JITEventListenerTest.cpp
  JITMemoryManagerTest.cpp
  JITTest.cpp
  MultiJITTest.cpp
  ${ProfileTestSources}
  )

if(MSVC)
  list(APPEND JITTestsSources JITTests.def)
endif()

add_llvm_unittest(JITTests
  ${JITTestsSources}
  )

if(MINGW OR CYGWIN)
  set_property(TARGET JITTests PROPERTY LINK_FLAGS -Wl,--export-all-symbols)
endif()

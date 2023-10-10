set(LLVM_LINK_COMPONENTS
  CAS
  Support
  )

set(SOURCES
  libCASPluginTest.cpp
  )

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/libCASPluginTest.exports)

add_llvm_library(CASPluginTest SHARED ${SOURCES})

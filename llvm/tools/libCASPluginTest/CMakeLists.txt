if (NOT LLVM_ENABLE_ONDISK_CAS)
  # The plugin is implemented on top of UnifiedOnDiskCache, which is only
  # functional when the on-disk CAS is enabled.
  return()
endif()

set(LLVM_LINK_COMPONENTS
  CAS
  Support
  )

set(SOURCES
  libCASPluginTest.cpp
  )

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/libCASPluginTest.exports)

add_llvm_library(CASPluginTest SHARED ${SOURCES})

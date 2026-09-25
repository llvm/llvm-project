# The plugin is implemented on top of UnifiedOnDiskCache, which is only
# functional when the on-disk CAS is enabled, and it is only useful as a shared
# library, which requires PIC objects.
if (NOT LLVM_ENABLE_ONDISK_CAS OR NOT LLVM_ENABLE_PIC)
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

# Only loaded by CAS unit tests out of the build tree.
add_llvm_library(CASPluginTest SHARED BUILDTREE_ONLY ${SOURCES}
  LINK_LIBS ${LLVM_PTHREAD_LIB}
)

if (NOT LLVM_ENABLE_ONDISK_CAS OR NOT LLVM_ENABLE_PIC)
  return()
endif()

set(LLVM_LINK_COMPONENTS
  CAS
  Support
  )

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/CASPluginTest.exports)

add_llvm_library(CASPluginTest SHARED BUILDTREE_ONLY
  CASPluginTest.cpp

  LINK_LIBS ${LLVM_PTHREAD_LIB}
  )

set_target_properties(CASPluginTest PROPERTIES
  PREFIX ""
  SUFFIX ${LLVM_PLUGIN_EXT}
  FOLDER "Tests"
  )
set_property(TARGET CASPluginTest PROPERTY VERSION)
set_property(TARGET CASPluginTest PROPERTY SOVERSION)

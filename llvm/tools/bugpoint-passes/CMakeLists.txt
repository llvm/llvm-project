if( NOT LLVM_BUILD_TOOLS )
  set(EXCLUDE_FROM_ALL ON)
endif()

# There's no reason to export anything from the plugin.
set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/bugpoint.exports)

if(WIN32 OR CYGWIN OR ZOS)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

add_llvm_library( BugpointPasses MODULE BUILDTREE_ONLY
  TestPasses.cpp

  DEPENDS
  intrinsics_gen
  bugpoint
  )

if (NOT WIN32 AND NOT CYGWIN)
  unset(LLVM_LINK_COMPONENTS)
  add_llvm_library(CGTestPlugin MODULE BUILDTREE_ONLY
    CodeGenTestPass.cpp
    Plugin.cpp
    )

  # Put PLUGIN next to the unit test executable.
  set_output_directory(CGTestPlugin
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/../
    LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/../
    )
  set_target_properties(CGTestPlugin PROPERTIES FOLDER "Tests")

  add_dependencies(CGTestPlugin intrinsics_gen vt_gen)
  add_dependencies(CGPluginTest CGTestPlugin)
endif ()

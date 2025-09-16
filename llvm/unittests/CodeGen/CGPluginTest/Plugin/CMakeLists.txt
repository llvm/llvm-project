if (NOT WIN32 AND NOT CYGWIN)
  unset(LLVM_LINK_COMPONENTS)
  add_llvm_library(CGTestPlugin MODULE BUILDTREE_ONLY
    CodeGenTestPass.cpp
    Plugin.cpp

    DEPENDS
    intrinsics_gen
    vt_gen
    )

  set_target_properties(CGTestPlugin PROPERTIES FOLDER "Tests")
  add_dependencies(CGPluginTest CGTestPlugin)
endif ()

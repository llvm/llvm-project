# Like Bye, the plugin expects the host process to provide the LLVM libraries.
# This doesn't work with DLLs on Windows.
if (NOT WIN32 AND NOT CYGWIN)
  # We don't use add_llvm_pass_plugin here, because it would make the plugin
  # subject to LLVM's static linking into tools feature. That's not viable here,
  # because the legacy registration path would inject the extra codegen-pass in
  # all tools unconditionally.
  add_llvm_example_library(Ciao MODULE BUILDTREE_ONLY
    Ciao.cpp

    DEPENDS
    intrinsics_gen
    vt_gen
    )
endif()

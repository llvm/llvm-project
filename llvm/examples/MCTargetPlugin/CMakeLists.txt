# The plugin expects to not link against the MC and Support libraries, but
# expects them to exist in the process loading the plugin. This doesn't work
# with DLLs on Windows (where a shared library can't have undefined
# references), so just skip this example on Windows.
if (NOT WIN32 AND NOT CYGWIN)
  add_llvm_example_library(MCTargetPlugin
    MODULE
    BUILDTREE_ONLY
    PLUGIN_TOOL
    llvm-mc

    MCTargetPlugin.cpp
    MCTargetPluginAsmParser.cpp
    )

  set_target_properties(MCTargetPlugin PROPERTIES FOLDER "Examples")
endif()

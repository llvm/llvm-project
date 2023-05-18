add_llvm_library(TestPlugin MODULE BUILDTREE_ONLY
    TestPlugin.cpp
    )

# Put PLUGIN next to the unit test executable.
set_output_directory(TestPlugin
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/../
    LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/../
    )
set_target_properties(TestPlugin PROPERTIES FOLDER "Tests")

# The plugin depends on some of the output files of intrinsics_gen, so make sure
# it is built before the plugin.
add_dependencies(TestPlugin intrinsics_gen)
add_dependencies(PluginsTests TestPlugin)

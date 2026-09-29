add_llvm_library(DoublerPlugin MODULE BUILDTREE_ONLY
    DoublerPlugin.cpp
    )

# Put PLUGIN next to the unit test executable.
set_output_directory(DoublerPlugin
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/../
    LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/../
    )
set_target_properties(DoublerPlugin PROPERTIES FOLDER "Tests")

# The plugin depends on some of the output files of intrinsics_gen, so make sure
# it is built before the plugin.
add_dependencies(DoublerPlugin intrinsics_gen)
add_dependencies(PluginsTests DoublerPlugin)

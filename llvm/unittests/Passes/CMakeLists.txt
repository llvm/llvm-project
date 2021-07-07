# Needed by LLVM's CMake checks because this file defines multiple targets.
set(LLVM_OPTIONAL_SOURCES PluginsTest.cpp TestPlugin.cpp DoublerPlugin.cpp PassBuilderBindingsTest.cpp)

# If plugins are disabled, this test will disable itself at runtime. Otherwise,
# reconfiguring with plugins disabled will leave behind a stale executable.
if (LLVM_ENABLE_PLUGINS)
  add_definitions(-DLLVM_ENABLE_PLUGINS)
endif()

# The plugin expects to not link against the Support and Core libraries,
# but expects them to exist in the process loading the plugin. This doesn't
# work with DLLs on Windows (where a shared library can't have undefined
# references), so just skip this testcase on Windows.
if (NOT WIN32)
  set(LLVM_LINK_COMPONENTS Support Passes Core AsmParser)
  add_llvm_unittest(PluginsTests
    PluginsTest.cpp
    )
  export_executable_symbols_for_plugins(PluginsTests)
  target_link_libraries(PluginsTests PRIVATE LLVMTestingSupport)

  set(LLVM_LINK_COMPONENTS)
  foreach(PLUGIN TestPlugin DoublerPlugin)
    add_llvm_library(${PLUGIN} MODULE BUILDTREE_ONLY ${PLUGIN}.cpp)

    # Put PLUGIN next to the unit test executable.
    set_output_directory(${PLUGIN}
      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
      LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
      )
    set_target_properties(${PLUGIN} PROPERTIES FOLDER "Tests")

    add_dependencies(${PLUGIN} intrinsics_gen)
    add_dependencies(PluginsTests ${PLUGIN})
  endforeach()

endif()

set(LLVM_LINK_COMPONENTS Support Passes Core Target native AllTargetsInfos)
add_llvm_unittest(PassesBindingsTests
	PassBuilderBindingsTest.cpp
	)
target_link_libraries(PassesBindingsTests PRIVATE LLVMTestingSupport)


# Needed by LLVM's CMake checks because this file defines multiple targets.
set(LLVM_OPTIONAL_SOURCES PluginsTest.cpp TestPlugin.cpp)

# If plugins are disabled, this test will disable itself at runtime. Otherwise,
# reconfiguring with plugins disabled will leave behind a stale executable.
if (LLVM_ENABLE_PLUGINS)
  add_definitions(-DLLVM_ENABLE_PLUGINS)
endif()

set(LLVM_LINK_COMPONENTS Support Passes Core)
add_llvm_unittest(PluginsTests
  PluginsTest.cpp
  )
export_executable_symbols(PluginsTests)

set(LLVM_LINK_COMPONENTS)
add_llvm_loadable_module(TestPlugin
  TestPlugin.cpp
  )

# Put plugin next to the unit test executable.
set_output_directory(TestPlugin
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
  LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
  )
set_target_properties(TestPlugin PROPERTIES FOLDER "Tests")

add_dependencies(TestPlugin intrinsics_gen)
add_dependencies(PluginsTests TestPlugin)

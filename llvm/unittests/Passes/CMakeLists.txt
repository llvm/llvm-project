set(LLVM_LINK_COMPONENTS Support Passes Core)

# If plugins are disabled, this test will disable itself at runtime. Otherwise,
# reconfiguring with plugins disabled will leave behind a stale executable.
if (LLVM_ENABLE_PLUGINS)
  add_definitions(-DLLVM_ENABLE_PLUGINS)
endif()

add_llvm_unittest(PluginsTests PluginsTest.cpp)
export_executable_symbols(PluginsTests)

add_library(TestPlugin MODULE TestPlugin.cxx)

set_output_directory(TestPlugin
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
  LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
  )

set_target_properties(TestPlugin
  PROPERTIES PREFIX ""
  SUFFIX ".so"
  )

if (WIN32 OR CYGWIN OR LLVM_EXPORT_SYMBOLS_FOR_PLUGINS)
  llvm_map_components_to_libnames(LLVM_DEPS ${LLVM_LINK_COMPONENTS})
  target_link_libraries(TestPlugin ${LLVM_DEPS})
endif()

add_dependencies(TestPlugin intrinsics_gen)
add_dependencies(PluginsTests TestPlugin)

# The plugin expects to not link against the Support and Core libraries,
# but expects them to exist in the process loading the plugin. This doesn't
# work with DLLs on Windows (where a shared library can't have undefined
# references), so just skip this testcase on Windows.
if (NOT WIN32 AND NOT CYGWIN)
  set(LLVM_LINK_COMPONENTS Support Passes Core AsmParser)
  add_llvm_unittest(PluginsTests
    PluginsTest.cpp

    EXPORT_SYMBOLS
    )
  target_link_libraries(PluginsTests PRIVATE LLVMTestingSupport)

  unset(LLVM_LINK_COMPONENTS)
  add_subdirectory(TestPlugin)
  add_subdirectory(DoublerPlugin)
endif()

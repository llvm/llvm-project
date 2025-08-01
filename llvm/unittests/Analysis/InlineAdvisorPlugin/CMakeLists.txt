# The advisor plugin expects to not link against the Analysis, Support and Core 
# libraries, but expects them to exist in the process loading the plugin. This 
# doesn't work with DLLs on Windows (where a shared library can't have undefined
# references), so just skip this testcase on Windows.
if ((NOT WIN32 AND NOT CYGWIN) OR LLVM_BUILD_LLVM_DYLIB)
  unset(LLVM_LINK_COMPONENTS)
  add_llvm_library(InlineAdvisorPlugin MODULE BUILDTREE_ONLY
    InlineAdvisorPlugin.cpp
    )
  # Put PLUGIN next to the unit test executable.
  set_output_directory(InlineAdvisorPlugin
      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/../
      LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/../
      )
  set_target_properties(InlineAdvisorPlugin PROPERTIES FOLDER "LLVM/Tests")

  # The plugin depends on some of the output files of intrinsics_gen, so make sure
  # it is built before the plugin.
  add_dependencies(InlineAdvisorPlugin intrinsics_gen)
  add_dependencies(AnalysisTests InlineAdvisorPlugin)
endif()

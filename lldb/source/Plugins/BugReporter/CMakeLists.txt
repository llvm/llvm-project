set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND BugReporter)

option(LLDB_ENABLE_GITHUB_BUG_REPORTER
  "Build the GitHub bug reporter so 'diagnostics report' can file a GitHub issue."
  ON)

# Reporters are tried in subdirectory order, so keep None (the fallback) last.
# Downstream adds its own reporter subdirectory ahead of these.
if(LLDB_ENABLE_GITHUB_BUG_REPORTER)
  add_subdirectory(GitHub)
endif()
add_subdirectory(None)

set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND BugReporter)

# Reporters are tried in subdirectory order, so keep None (the fallback) last.
# Downstream adds its own reporter subdirectory ahead of these.
add_subdirectory(None)

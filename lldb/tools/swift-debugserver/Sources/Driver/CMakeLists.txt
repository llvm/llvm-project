add_lldb_tool(swift-debugserver ADD_TO_FRAMEWORK
  DebugServer.swift
  Version.swift
)

set_target_properties(swift-debugserver PROPERTIES Swift_MODULE_NAME "swift_debugserver")

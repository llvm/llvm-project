add_lldb_library(lldbServerPluginMockAccelerator
  LLDBServerMockAcceleratorPlugin.cpp

  LINK_LIBS
    lldbHost
    lldbUtility
)

target_include_directories(lldbServerPluginMockAccelerator
  PRIVATE
    "${LLDB_SOURCE_DIR}/source"
)

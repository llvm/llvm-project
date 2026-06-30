add_lldb_library(lldbServerPluginMockAccelerator
  LLDBServerMockAcceleratorPlugin.cpp
  ProcessMockAccelerator.cpp
  ThreadMockAccelerator.cpp
  RegisterContextMockAccelerator.cpp

  LINK_LIBS
    lldbHost
    lldbUtility
)

target_include_directories(lldbServerPluginMockAccelerator
  PRIVATE
    "${LLDB_SOURCE_DIR}/source"
)

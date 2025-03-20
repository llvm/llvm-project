
add_lldb_library(lldbServerPluginMockGPU SHARED
  MockGPUPlugin.cpp
)

target_link_libraries(lldbServerPluginMockGPU lldbServerPluginInterface)

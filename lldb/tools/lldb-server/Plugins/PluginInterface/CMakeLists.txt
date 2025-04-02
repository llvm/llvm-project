
add_lldb_library(lldbServerPluginInterface
  LLDBServerNativeProcess.cpp
  LLDBServerPlugin.cpp
)

target_link_libraries(lldbServerPluginInterface lldbServerPluginMockGPU)
target_include_directories(lldbServerPluginInterface PRIVATE "${LLDB_SOURCE_DIR}/source")

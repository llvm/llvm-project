
add_lldb_library(lldbServerPluginMockGPU
  LLDBServerPluginMockGPU.cpp
  ProcessMockGPU.cpp
  RegisterContextMockGPU.cpp
  ThreadMockGPU.cpp
)

target_include_directories(lldbServerPluginMockGPU PRIVATE "${LLDB_SOURCE_DIR}/source" "../..")

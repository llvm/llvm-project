add_lldb_library(lldbServerPluginMockAccelerator
  LLDBServerPluginMockAccelerator.cpp
  ../LLDBServerPluginAccelerator.cpp
)

target_include_directories(lldbServerPluginMockAccelerator
  PRIVATE
    "${LLDB_SOURCE_DIR}/source"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

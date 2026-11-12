message(STATUS "Building the llvm-offload-device-info tool")

add_offload_tool(llvm-offload-device-info llvm-offload-device-info.cpp)

target_link_libraries(llvm-offload-device-info PRIVATE
  LLVMOffload
)

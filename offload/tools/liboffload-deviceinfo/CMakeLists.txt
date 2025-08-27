message(STATUS "Building the llvm-liboffload-device-info tool")

add_openmp_tool(llvm-liboffload-device-info llvm-liboffload-device-info.cpp)

llvm_update_compile_flags(llvm-liboffload-device-info)

target_link_libraries(llvm-liboffload-device-info PRIVATE
  LLVMOffload
)

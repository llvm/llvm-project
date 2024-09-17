message(STATUS "Building the llvm-offload-device-info tool")

add_openmp_tool(llvm-offload-device-info llvm-offload-device-info.cpp)

llvm_update_compile_flags(llvm-offload-device-info)

target_include_directories(llvm-offload-device-info PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
)
target_link_libraries(llvm-offload-device-info PRIVATE
  omp
  omptarget
)

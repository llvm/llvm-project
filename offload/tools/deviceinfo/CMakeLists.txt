message(STATUS "Building the llvm-omp-device-info tool")

add_openmp_tool(llvm-omp-device-info llvm-omp-device-info.cpp)

llvm_update_compile_flags(llvm-omp-device-info)

target_include_directories(llvm-omp-device-info PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
)
target_link_libraries(llvm-omp-device-info PRIVATE
  omp
  omptarget
)

message(STATUS "Building the llvm-omp-kernel-replay tool")

add_openmp_tool(llvm-omp-kernel-replay llvm-omp-kernel-replay.cpp)

llvm_update_compile_flags(llvm-omp-kernel-replay)

target_include_directories(llvm-omp-kernel-replay PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
)
target_link_libraries(llvm-omp-kernel-replay PRIVATE
  LLVMSupport
  omp
  omptarget
)

set(CUDA_HOST_COMPILER gcc)
set(CUDA_NVCC_FLAGS -std=c++11)

if(ACXXEL_ENABLE_CUDA)
cuda_add_executable(simple_example simple_example.cu)
target_link_libraries(simple_example acxxel)
endif()

if(ACXXEL_ENABLE_OPENCL)
add_executable(opencl_example opencl_example.cpp)
target_link_libraries(opencl_example acxxel ${OpenCL_LIBRARIES})
endif()

if(STREAM_EXECUTOR_ENABLE_CUDA_PLATFORM)
    set(CUDA_TEST_SOURCES cuda/CUDATest.cpp)
endif()

add_se_unittest(
  CoreTests
  DeviceTest.cpp
  KernelSpecTest.cpp
  PackedKernelArgumentArrayTest.cpp
  StreamTest.cpp
  ${CUDA_TEST_SOURCES}
)

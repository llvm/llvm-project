add_offload_test_device_code(CUDAMath.cpp cuda-math WITH_DEVICE_MATH_LIBS -O3 -stdlib -fno-builtin)
add_offload_test_device_code(HIPMath.cpp hip-math WITH_DEVICE_MATH_LIBS -O3 -stdlib -fno-builtin)
add_offload_test_device_code(LLVMLibm.cpp llvm-libm -O3 -stdlib -fno-builtin)

add_custom_target(conformance_device_binaries DEPENDS
  cuda-math.bin
  hip-math.bin
  llvm-libm.bin
)
set(OFFLOAD_CONFORMANCE_DEVICE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

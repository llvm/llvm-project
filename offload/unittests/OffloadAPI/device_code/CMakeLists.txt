add_offload_test_device_code(foo.c foo)
add_offload_test_device_code(bar.c bar)
# Compile with optimizations to eliminate AMDGPU implicit arguments.
add_offload_test_device_code(noargs.c noargs -O3)

add_custom_target(offload_device_binaries DEPENDS foo.bin bar.bin noargs.bin)
set(OFFLOAD_TEST_DEVICE_CODE_PATH ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

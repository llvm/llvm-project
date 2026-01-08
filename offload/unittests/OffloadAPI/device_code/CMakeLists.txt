add_offload_test_device_code(foo.cpp foo)
add_offload_test_device_code(bar.cpp bar)
# Compile with optimizations to eliminate AMDGPU implicit arguments.
add_offload_test_device_code(noargs.cpp noargs -O3)
add_offload_test_device_code(byte.cpp byte)
add_offload_test_device_code(localmem.cpp localmem)
add_offload_test_device_code(localmem_reduction.cpp localmem_reduction)
add_offload_test_device_code(localmem_static.cpp localmem_static)
add_offload_test_device_code(global.cpp global)
add_offload_test_device_code(global_ctor.cpp global_ctor)
add_offload_test_device_code(global_dtor.cpp global_dtor)
add_offload_test_device_code(sequence.cpp sequence)

add_custom_target(offload_device_binaries DEPENDS
    foo.bin
    bar.bin
    noargs.bin
    byte.bin
    localmem.bin
    localmem_reduction.bin
    localmem_static.bin
    global.bin
    global_ctor.bin
    global_dtor.bin
    sequence.bin
)
set(OFFLOAD_TEST_DEVICE_CODE_PATH ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

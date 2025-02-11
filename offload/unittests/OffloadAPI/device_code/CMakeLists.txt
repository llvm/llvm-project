macro(add_offload_test_device_code test_filename test_name)
    message("Building Offload API device code for test '${test_name}'")
    set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${test_filename})
    set(UTIL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/util.h)

    # Build for NVPTX
    set(BIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.nvptx64.bin)
    add_custom_command(OUTPUT ${BIN_PATH}
        COMMAND
        ${CMAKE_C_COMPILER} --target=nvptx64-nvidia-cuda -march=native
        --cuda-path=/usr/local/cuda
        ${SRC_PATH} -o ${BIN_PATH}
        DEPENDS ${SRC_PATH} ${UTIL_PATH}
    )
    list(APPEND BIN_PATHS ${BIN_PATH})

    # Build for AMDGPU
    set(BIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.amdgpu.bin)
    add_custom_command(OUTPUT ${BIN_PATH}
        COMMAND
        ${CMAKE_C_COMPILER} --target=amdgcn-amd-amdhsa -nogpulib
        ${SRC_PATH} -o ${BIN_PATH}
        DEPENDS ${SRC_PATH} ${UTIL_PATH}
    )
    list(APPEND BIN_PATHS ${BIN_PATH})

    # TODO: Build for host CPU
endmacro()


add_offload_test_device_code(foo.c foo)
add_offload_test_device_code(bar.c bar)

add_custom_target(LibompUnitTestsDeviceBins DEPENDS ${BIN_PATHS})

set(OFFLOAD_TEST_DEVICE_CODE_PATH ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

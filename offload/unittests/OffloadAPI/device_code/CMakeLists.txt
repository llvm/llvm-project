macro(add_offload_test_device_code test_filename test_name)
    set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${test_filename})

    # Build for NVPTX
    if(OFFLOAD_TEST_TARGET_NVIDIA)
        set(BIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.nvptx64.bin)
        add_custom_command(OUTPUT ${BIN_PATH}
            COMMAND
            ${CMAKE_C_COMPILER} --target=nvptx64-nvidia-cuda
            -march=${LIBOMPTARGET_DEP_CUDA_ARCH}
            --cuda-path=${CUDA_ROOT}
            ${SRC_PATH} -o ${BIN_PATH}
            DEPENDS ${SRC_PATH}
        )
        list(APPEND BIN_PATHS ${BIN_PATH})
    endif()

    # Build for AMDGPU
    if(OFFLOAD_TEST_TARGET_AMDGPU)
        set(BIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.amdgpu.bin)
        add_custom_command(OUTPUT ${BIN_PATH}
            COMMAND
            ${CMAKE_C_COMPILER} --target=amdgcn-amd-amdhsa -nogpulib
            -mcpu=${LIBOMPTARGET_DEP_AMDGPU_ARCH}
            ${SRC_PATH} -o ${BIN_PATH}
            DEPENDS ${SRC_PATH}
        )
        list(APPEND BIN_PATHS ${BIN_PATH})
    endif()

    # TODO: Build for host CPU
endmacro()


# Decide what device targets to build for. LibomptargetGetDependencies is
# included at the top-level so the GPUs present on the system are already
# detected.
set(OFFLOAD_TESTS_FORCE_NVIDIA_ARCH "" CACHE STRING
    "Force building of NVPTX device code for Offload unit tests with the given arch, e.g. sm_61")
set(OFFLOAD_TESTS_FORCE_AMDGPU_ARCH "" CACHE STRING
    "Force building of AMDGPU device code for Offload unit tests with the given arch, e.g. gfx1030")

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  get_filename_component(CUDA_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY ABSOLUTE)
endif()
if (OFFLOAD_TESTS_FORCE_NVIDIA_ARCH)
    set(LIBOMPTARGET_DEP_CUDA_ARCH ${OFFLOAD_TESTS_FORCE_NVIDIA_ARCH})
    set(OFFLOAD_TEST_TARGET_NVIDIA ON)
elseif (LIBOMPTARGET_FOUND_NVIDIA_GPU AND CUDA_ROOT AND "cuda" IN_LIST LIBOMPTARGET_PLUGINS_TO_BUILD)
    set(OFFLOAD_TEST_TARGET_NVIDIA ON)
endif()

if (OFFLOAD_TESTS_FORCE_AMDGPU_ARCH)
    set(LIBOMPTARGET_DEP_AMDGPU_ARCH ${OFFLOAD_TESTS_FORCE_AMDGPU_ARCH})
    set(OFFLOAD_TEST_TARGET_AMDGPU ON)
elseif (LIBOMPTARGET_FOUND_AMDGPU_GPU AND "amdgpu" IN_LIST LIBOMPTARGET_PLUGINS_TO_BUILD)
    list(GET LIBOMPTARGET_AMDGPU_DETECTED_ARCH_LIST 0 LIBOMPTARGET_DEP_AMDGPU_ARCH)
    set(OFFLOAD_TEST_TARGET_AMDGPU ON)
endif()

add_offload_test_device_code(foo.c foo)
add_offload_test_device_code(bar.c bar)

add_custom_target(OffloadUnitTestsDeviceBins DEPENDS ${BIN_PATHS})

set(OFFLOAD_TEST_DEVICE_CODE_PATH ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

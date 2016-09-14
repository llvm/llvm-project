macro(add_se_library name)
  add_llvm_library(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "streamexecutor libraries")
endmacro(add_se_library)

if(STREAM_EXECUTOR_ENABLE_CUDA_PLATFORM)
    set(
        CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cuda/cmake/modules/")

    find_package(Libcuda REQUIRED)
    include_directories(${LIBCUDA_INCLUDE_DIRS})

    set(
        STREAM_EXECUTOR_CUDA_PLATFORM_TARGET_OBJECT
        $<TARGET_OBJECTS:streamexecutor_cuda_platform>)

    set(
        STREAM_EXECUTOR_LIBCUDA_LIBRARIES
        ${LIBCUDA_LIBRARIES})
endif(STREAM_EXECUTOR_ENABLE_CUDA_PLATFORM)

add_subdirectory(platforms)

add_se_library(
    streamexecutor
    Device.cpp
    DeviceMemory.cpp
    Error.cpp
    HostMemory.cpp
    Kernel.cpp
    KernelSpec.cpp
    PackedKernelArgumentArray.cpp
    Platform.cpp
    PlatformDevice.cpp
    PlatformManager.cpp
    Stream.cpp
    ${STREAM_EXECUTOR_CUDA_PLATFORM_TARGET_OBJECT}
    LINK_LIBS
    ${STREAM_EXECUTOR_LIBCUDA_LIBRARIES})

install(TARGETS streamexecutor DESTINATION lib)

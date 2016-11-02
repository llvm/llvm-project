macro(add_se_library name)
  add_llvm_library(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "streamexecutor libraries")
endmacro(add_se_library)

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

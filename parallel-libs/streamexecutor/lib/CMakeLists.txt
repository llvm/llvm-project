add_library(
    utils
    OBJECT
    Utils/Error.cpp)

add_library(
    streamexecutor
    $<TARGET_OBJECTS:utils>
    Device.cpp
    DeviceMemory.cpp
    Kernel.cpp
    KernelSpec.cpp
    PackedKernelArgumentArray.cpp
    Platform.cpp
    PlatformDevice.cpp
    PlatformManager.cpp
    Stream.cpp)
target_link_libraries(streamexecutor ${llvm_libs})

install(TARGETS streamexecutor DESTINATION lib)

if(STREAM_EXECUTOR_UNIT_TESTS)
    add_subdirectory(unittests)
endif()

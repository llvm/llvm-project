add_library(
    utils
    OBJECT
    Utils/Error.cpp)

add_library(
    streamexecutor
    $<TARGET_OBJECTS:utils>
    Device.cpp
    Kernel.cpp
    KernelSpec.cpp
    PackedKernelArgumentArray.cpp
    PlatformInterfaces.cpp
    Stream.cpp)
target_link_libraries(streamexecutor ${llvm_libs})

if(STREAM_EXECUTOR_UNIT_TESTS)
    add_subdirectory(unittests)
endif()

set(PLUGINS_TEST_COMMON LLVMOffload)
set(PLUGINS_TEST_INCLUDE ${LIBOMPTARGET_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(device_code)

add_offload_unittest("device"
    device/olIterateDevices.cpp
    device/olGetDeviceInfo.cpp
    device/olGetDeviceInfoSize.cpp)

add_offload_unittest("event"
    event/olDestroyEvent.cpp
    event/olWaitEvent.cpp)

add_offload_unittest("init"
    init/olInit.cpp)
target_compile_definitions("init.unittests" PRIVATE DISABLE_WRAPPER)

add_offload_unittest("kernel"
    kernel/olGetKernel.cpp
    kernel/olLaunchKernel.cpp)

add_offload_unittest("memory"
    memory/olMemAlloc.cpp
    memory/olMemFree.cpp
    memory/olMemcpy.cpp)

add_offload_unittest("platform"
    platform/olGetPlatformInfo.cpp
    platform/olGetPlatformInfoSize.cpp)

add_offload_unittest("program"
    program/olCreateProgram.cpp
    program/olDestroyProgram.cpp)

add_offload_unittest("queue"
    queue/olCreateQueue.cpp
    queue/olWaitQueue.cpp
    queue/olDestroyQueue.cpp)

set(PLUGINS_TEST_COMMON LLVMOffload)
set(PLUGINS_TEST_INCLUDE ${LIBOMPTARGET_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(device_code)

add_offload_unittest("device"
    device/olIterateDevices.cpp
    device/olGetDeviceInfo.cpp
    device/olGetDeviceInfoSize.cpp)

add_offload_unittest("event"
    event/olCreateEvent.cpp
    event/olDestroyEvent.cpp
    event/olSyncEvent.cpp
    event/olGetEventInfo.cpp
    event/olGetEventInfoSize.cpp)

add_offload_unittest("init"
    init/olInit.cpp)
target_compile_definitions("init.unittests" PRIVATE DISABLE_WRAPPER)

add_offload_unittest("kernel"
    kernel/olCalculateOptimalOccupancy.cpp
    kernel/olLaunchKernel.cpp)

add_offload_unittest("memory"
    memory/olMemAlloc.cpp
    memory/olMemFill.cpp
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
    queue/olSyncQueue.cpp
    queue/olDestroyQueue.cpp
    queue/olGetQueueInfo.cpp
    queue/olGetQueueInfoSize.cpp
    queue/olWaitEvents.cpp
    queue/olLaunchHostFunction.cpp)

add_offload_unittest("symbol"
    symbol/olGetSymbol.cpp
    symbol/olGetSymbolInfo.cpp
    symbol/olGetSymbolInfoSize.cpp)

set(PLUGINS_TEST_COMMON LLVMOffload)
set(PLUGINS_TEST_INCLUDE ${LIBOMPTARGET_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(device_code)
message(${OFFLOAD_TEST_DEVICE_CODE_PATH})

add_libompt_unittest("offload.unittests"
    ${CMAKE_CURRENT_SOURCE_DIR}/common/Environment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform/olGetPlatform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform/olGetPlatformCount.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform/olGetPlatformInfo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform/olGetPlatformInfoSize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/olGetDevice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/olGetDeviceCount.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/olGetDeviceInfo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/olGetDeviceInfoSize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue/olCreateQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue/olWaitQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue/olReleaseQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue/olRetainQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/olMemAlloc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/olMemFree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue/olEnqueueDataWrite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue/olEnqueueDataRead.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue/olEnqueueDataCopy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue/olEnqueueMemcpy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue/olEnqueueKernelLaunch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/olCreateProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/olRetainProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/olReleaseProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/olCreateKernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/olReleaseKernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/olRetainKernel.cpp
    )
add_dependencies("offload.unittests" ${PLUGINS_TEST_COMMON} LibomptUnitTestsDeviceBins)
target_compile_definitions("offload.unittests" PRIVATE DEVICE_CODE_PATH="${OFFLOAD_TEST_DEVICE_CODE_PATH}")
target_link_libraries("offload.unittests" PRIVATE ${PLUGINS_TEST_COMMON})
target_include_directories("offload.unittests" PRIVATE ${PLUGINS_TEST_INCLUDE})

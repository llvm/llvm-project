set(PLUGINS_TEST_COMMON LLVMOffload)
set(PLUGINS_TEST_INCLUDE ${LIBOMPTARGET_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/queue/olFinishQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue/olReleaseQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue/olRetainQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/olMemAlloc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/olMemFree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue/olEnqueueDataWrite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue/olEnqueueDataRead.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue/olEnqueueDataCopy.cpp
    )
add_dependencies("offload.unittests" ${PLUGINS_TEST_COMMON})
target_link_libraries("offload.unittests" PRIVATE ${PLUGINS_TEST_COMMON})
target_include_directories("offload.unittests" PRIVATE ${PLUGINS_TEST_INCLUDE})

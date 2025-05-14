set(PLUGINS_TEST_COMMON LLVMOffload)
set(PLUGINS_TEST_INCLUDE ${LIBOMPTARGET_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(device_code)

add_offload_unittest("offload.unittests"
    ${CMAKE_CURRENT_SOURCE_DIR}/common/Environment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform/olGetPlatformInfo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform/olGetPlatformInfoSize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/olIterateDevices.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/olGetDeviceInfo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/olGetDeviceInfoSize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue/olCreateQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue/olWaitQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue/olDestroyQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/olMemAlloc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/olMemFree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/olMemcpy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/olCreateProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/olDestroyProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/olGetKernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/olLaunchKernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event/olDestroyEvent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event/olWaitEvent.cpp
    )
add_dependencies("offload.unittests" ${PLUGINS_TEST_COMMON} OffloadUnitTestsDeviceBins)
target_compile_definitions("offload.unittests" PRIVATE DEVICE_CODE_PATH="${OFFLOAD_TEST_DEVICE_CODE_PATH}")
target_link_libraries("offload.unittests" PRIVATE ${PLUGINS_TEST_COMMON})
target_include_directories("offload.unittests" PRIVATE ${PLUGINS_TEST_INCLUDE})

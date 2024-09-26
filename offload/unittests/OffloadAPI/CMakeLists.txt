set(PLUGINS_TEST_COMMON offload_new)
set(PLUGINS_TEST_INCLUDE ${LIBOMPTARGET_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_libompt_unittest("offload.unittests"
    ${CMAKE_CURRENT_SOURCE_DIR}/common/environment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform/offloadPlatformGet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform/offloadPlatformGetInfo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/offloadDeviceGet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/offloadDeviceGetInfo.cpp)
add_dependencies("offload.unittests" ${PLUGINS_TEST_COMMON})
target_link_libraries("offload.unittests" PRIVATE ${PLUGINS_TEST_COMMON})
target_include_directories("offload.unittests" PRIVATE ${PLUGINS_TEST_INCLUDE})

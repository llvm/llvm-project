add_custom_target(offload.conformance)

set(PLUGINS_TEST_COMMON LLVMOffload LLVMSupport)
set(PLUGINS_TEST_INCLUDE ${LIBOMPTARGET_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(device_code)

add_conformance_test(sin sin.cpp)

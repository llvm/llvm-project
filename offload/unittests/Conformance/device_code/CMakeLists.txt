# FIXME: Currently missing dependencies to build GPU portion automatically.
add_offload_test_device_code(sin.c sin)

set(OFFLOAD_TEST_DEVICE_CODE_PATH ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

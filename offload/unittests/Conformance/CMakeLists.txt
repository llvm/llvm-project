add_custom_target(offload.conformance)

set(PLUGINS_TEST_COMMON MathTest)

add_subdirectory(device_code)
add_subdirectory(lib)
add_subdirectory(tests)

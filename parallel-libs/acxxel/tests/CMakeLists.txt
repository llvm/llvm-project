add_executable(acxxel_test acxxel_test.cpp)
target_link_libraries(
    acxxel_test
    acxxel
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})
add_test(AcxxelTest acxxel_test)

add_executable(span_test span_test.cpp)
target_link_libraries(
    span_test
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})
add_test(SpanTest span_test)

add_executable(status_test status_test.cpp)
target_link_libraries(
    status_test
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})
add_test(StatusTest status_test)

if(ACXXEL_ENABLE_OPENCL)
add_executable(opencl_test opencl_test.cpp)
target_link_libraries(
    opencl_test
    acxxel
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})
add_test(OpenCLTest opencl_test)
endif()

if(ACXXEL_ENABLE_MULTI_DEVICE_UNIT_TESTS)
add_executable(multi_device_test multi_device_test.cpp)
target_link_libraries(
    multi_device_test
    acxxel
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})
add_test(MultiDeviceTest multi_device_test)
endif()

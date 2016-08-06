add_executable(
    kernel_test
    KernelTest.cpp)
target_link_libraries(
    kernel_test
    streamexecutor
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})
add_test(KernelTest kernel_test)

add_executable(
    kernel_spec_test
    KernelSpecTest.cpp)
target_link_libraries(
    kernel_spec_test
    streamexecutor
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})
add_test(KernelSpecTest kernel_spec_test)

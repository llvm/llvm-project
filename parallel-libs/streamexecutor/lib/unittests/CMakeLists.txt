add_executable(
    device_test
    DeviceTest.cpp)
target_link_libraries(
    device_test
    streamexecutor
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})
add_test(DeviceTest device_test)

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

add_executable(
    packed_kernel_argument_array_test
    PackedKernelArgumentArrayTest.cpp)
target_link_libraries(
    packed_kernel_argument_array_test
    streamexecutor
    ${llvm_libs}
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})
add_test(PackedKernelArgumentArrayTest packed_kernel_argument_array_test)

add_executable(
    stream_test
    StreamTest.cpp)
target_link_libraries(
    stream_test
    streamexecutor
    ${llvm_libs}
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})
add_test(StreamTest stream_test)

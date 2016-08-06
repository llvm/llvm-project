add_library(
    utils
    OBJECT
    Utils/Error.cpp)

add_library(
    streamexecutor
    $<TARGET_OBJECTS:utils>
    Kernel.cpp
    KernelSpec.cpp)
target_link_libraries(streamexecutor ${llvm_libs})

if(STREAM_EXECUTOR_UNIT_TESTS)
    add_subdirectory(unittests)
endif()

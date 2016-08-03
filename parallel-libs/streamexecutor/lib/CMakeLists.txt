add_library(
    utils
    OBJECT
    Utils/Error.cpp)

add_library(
    streamexecutor
    $<TARGET_OBJECTS:utils>
    KernelSpec.cpp)

if(STREAM_EXECUTOR_UNIT_TESTS)
    add_subdirectory(unittests)
endif()

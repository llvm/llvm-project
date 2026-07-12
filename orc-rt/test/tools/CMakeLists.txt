# Tools used by the ORC-RT regression tests. These are test-support binaries
# built into the test tree; they are not installed with the runtime.

# --- smoke-check test tool ---
add_executable(orc-rt-smoke-check orc-rt-smoke-check.cpp)

# -- logging-check test tool ---
add_executable(orc-rt-log-check orc-rt-log-check.cpp)
target_compile_options(orc-rt-log-check PRIVATE ${ORC_RT_COMPILE_FLAGS})
target_link_libraries(orc-rt-log-check PRIVATE orc-rt-executor)

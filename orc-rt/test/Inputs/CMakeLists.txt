# Inputs shared by the unit and regression tests.

# A minimal shared library, for tests that load, search, and close a host
# library. Unit tests find it through the ORC_RT_TEST_DYLIB_PATH compile
# definition, and regression tests through the %{test-dylib} substitution.
add_library(orc-rt-test-dylib SHARED orc-rt-test-dylib.c)

# Build it at a path known at configure time, so that it can be written into
# the regression tests' lit.site.cfg.py. Wrapping the directory in a generator
# expression stops multi-config generators from appending a per-config
# subdirectory. RUNTIME_OUTPUT_DIRECTORY places the DLL on Windows.
set(ORC_RT_TEST_DYLIB_DIR ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(orc-rt-test-dylib PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "$<1:${ORC_RT_TEST_DYLIB_DIR}>"
  RUNTIME_OUTPUT_DIRECTORY "$<1:${ORC_RT_TEST_DYLIB_DIR}>"
  FOLDER "orc-rt/Tests")

set(ORC_RT_TEST_DYLIB_PATH
  "${ORC_RT_TEST_DYLIB_DIR}/orc-rt-test-dylib${CMAKE_SHARED_LIBRARY_SUFFIX}"
  PARENT_SCOPE)

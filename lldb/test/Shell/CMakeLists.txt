# Configure the Shell test suite.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit-lldb-init.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit-lldb-init)

if (CMAKE_GENERATOR STREQUAL "Xcode")
  # Xcode does not get the auto-generated targets. We need to create
  # check-lldb-shell manually.
  add_lit_testsuite(check-lldb-shell "Running lldb shell test suite"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS lldb-test-deps)
endif()

# Add a lit test suite that runs the shell test while capturing a reproducer.
add_lit_testsuite(check-lldb-repro-capture
  "Running lldb shell test suite with reproducer capture"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS "lldb-run-with-repro=capture"
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-test-deps)

# Add a lit test suite that runs the shell test by replaying a reproducer.
add_lit_testsuite(check-lldb-repro
  "Running lldb shell test suite with reproducer replay"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS "lldb-run-with-repro=replay"
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-test-deps)
add_dependencies(check-lldb-repro check-lldb-repro-capture)

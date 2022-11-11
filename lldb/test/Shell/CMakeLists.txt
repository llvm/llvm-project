add_custom_target(lldb-shell-test-deps)
add_dependencies(lldb-shell-test-deps lldb-test-depends)

add_lit_testsuites(LLDB-SHELL
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS lldb-shell-test-deps)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit-lldb-init.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit-lldb-init)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit-lldb-init-quiet
  "command source -C --silent-run true lit-lldb-init\n")

add_lit_testsuite(check-lldb-shell "Running lldb shell test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-shell-test-deps)

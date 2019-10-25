# Configure the Shell test suite.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit-lldb-init.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit-lldb-init)

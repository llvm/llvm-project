configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_lit_testsuite(check-llgo "Running the llgo regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    FileCheck
    count
    llgo
    llgoi
    libgo
    not
  )
set_target_properties(check-llgo PROPERTIES FOLDER "Tests")

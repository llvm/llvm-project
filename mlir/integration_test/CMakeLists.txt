# Passed to lit.site.cfg.py.in to set up the path where to find the libraries.
set(MLIR_INTEGRATION_TEST_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(MLIR_INTEGRATION_TEST_DEPENDS
  FileCheck count not
  mlir-cpu-runner
  mlir-opt
  mlir_runner_utils
  mlir_c_runner_utils
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

add_lit_testsuite(check-mlir-integration "Running the MLIR integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_INTEGRATION_TEST_DEPENDS}
  )
set_target_properties(check-mlir-integration PROPERTIES FOLDER "integration_test")

add_lit_testsuites(MLIR_INTEGRATION ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_INTEGRATION_TEST_DEPENDS}
  )

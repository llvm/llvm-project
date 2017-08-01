# The configured file is not placed in the correct location
# until the tests are run as we need to copy it into
# a copy of the tests folder
configure_file("lit.site.cfg.in" "lit.site.cfg" @ONLY)

# For every lit.cfg in the Inputs tree, create a lit.site.cfg that points at
# the tests back in the source directory. Running the tests this way keeps the
# temporary output files in the build directory.
file(GLOB inputs_suites "Inputs/*/lit.cfg")
foreach(lit_cfg ${inputs_suites})
  get_filename_component(LIT_TEST_SRC_DIR "${lit_cfg}" DIRECTORY)
  file(RELATIVE_PATH LIT_TEST_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}" "${LIT_TEST_SRC_DIR}")
  set(LIT_TEST_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/${LIT_TEST_BIN_DIR}")
  configure_file("Inputs/lit.site.cfg.in" "${LIT_TEST_BIN_DIR}/lit.site.cfg" @ONLY)
endforeach()

# Add rules for lit's own test suite
add_lit_testsuite(check-lit "Running lit's tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "FileCheck" "not"
)

# For IDEs
set_target_properties(check-lit PROPERTIES FOLDER "Tests")

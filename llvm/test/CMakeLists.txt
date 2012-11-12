configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

# Don't include check-llvm into check-all without LLVM_BUILD_TOOLS.
if(NOT LLVM_BUILD_TOOLS)
  set(EXCLUDE_FROM_ALL ON)
endif()

add_lit_testsuite(check-llvm "Running the LLVM regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS llvm_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
         llvm_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  DEPENDS UnitTests
          BugpointPasses LLVMHello
          llc lli llvm-ar llvm-as
          llvm-bcanalyzer llvm-diff
          llvm-dis llvm-extract llvm-dwarfdump
          llvm-link
          llvm-mc
          llvm-mcmarkup
          llvm-nm
          llvm-objdump
          llvm-readobj
          macho-dump opt
          profile_rt-shared
          FileCheck count not
          yaml2obj
  )
set_target_properties(check-llvm PROPERTIES FOLDER "Tests")

# Setup a legacy alias for 'check-llvm'. This will likely change to be an
# alias for 'check-all' at some point in the future.
add_custom_target(check)
add_dependencies(check check-llvm)
set_target_properties(check PROPERTIES FOLDER "Tests")

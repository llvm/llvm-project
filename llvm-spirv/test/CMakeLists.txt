llvm_canonicalize_cmake_booleans(SKIP_SPIRV_DEBUG_INFO_TESTS)

# required by lit.site.cfg.py.in
get_target_property(LLVM_SPIRV_DIR llvm-spirv BINARY_DIR)
set(LLVM_SPIRV_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

if(NOT BUILD_EXTERNAL)
  set(LLVM_SPIRV_TEST_DEPS
    FileCheck
    count
    llvm-as
    llvm-dis
    not
  )
  if(NOT SKIP_SPIRV_DEBUG_INFO_TESTS)
    list(APPEND LLVM_SPIRV_TEST_DEPS
      llc
      llvm-dwarfdump
      llvm-objdump
      llvm-readobj
    )
  endif(NOT SKIP_SPIRV_DEBUG_INFO_TESTS)
endif(NOT BUILD_EXTERNAL)


add_lit_testsuite(check-llvm-spirv "Running the LLVM-SPIRV regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS
    --verbose
  DEPENDS
    ${LLVM_SPIRV_TEST_DEPS}
    llvm-spirv
)

# to enable a custom test target on cmake below 3.11
# starting with 3.11 "test" is only reserved if ENABLE_TESTING(ON)
if(BUILD_EXTERNAL)
  cmake_policy(PUSH)
  if(POLICY CMP0037 AND ${CMAKE_VERSION} VERSION_LESS "3.11.0")
    cmake_policy(SET CMP0037 OLD)
  endif(POLICY CMP0037 AND ${CMAKE_VERSION} VERSION_LESS "3.11.0")
  add_custom_target(test
    DEPENDS
      check-llvm-spirv
  )
  cmake_policy(POP)
endif(BUILD_EXTERNAL)

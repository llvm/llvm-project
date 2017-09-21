llvm_canonicalize_cmake_booleans(
  LLVM_TOOL_LTO_BUILD
  HAVE_OCAMLOPT
  HAVE_OCAML_OUNIT
  LLVM_INCLUDE_GO_TESTS
  LLVM_USE_INTEL_JITEVENTS
  HAVE_LIBZ
  HAVE_LIBXAR
  LLVM_ENABLE_DIA_SDK
  LLVM_ENABLE_FFI
  BUILD_SHARED_LIBS)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

# Don't include check-llvm into check-all without LLVM_BUILD_TOOLS.
if(NOT LLVM_BUILD_TOOLS)
  set(EXCLUDE_FROM_ALL ON)
endif()

# Set the depends list as a variable so that it can grow conditionally.
# NOTE: Sync the substitutions in test/lit.cfg when adding to this list.
set(LLVM_TEST_DEPENDS
          BugpointPasses
          FileCheck
          LLVMHello
          UnitTests
          bugpoint
          count
          llc
          lli
          lli-child-target
          llvm-ar
          llvm-as
          llvm-bcanalyzer
          llvm-c-test
          llvm-cat
          llvm-cxxfilt
          llvm-config
          llvm-cov
          llvm-cxxdump
          llvm-cvtres
          llvm-diff
          llvm-dis
          llvm-dlltool
          llvm-dsymutil
          llvm-dwarfdump
          llvm-dwp
          llvm-extract
          llvm-isel-fuzzer
          llvm-lib
          llvm-link
          llvm-lto2
          llvm-mc
          llvm-mcmarkup
          llvm-modextract
          llvm-mt
          llvm-nm
          llvm-objcopy
          llvm-objdump
          llvm-opt-report
          llvm-pdbutil
          llvm-profdata
          llvm-ranlib
          llvm-rc
          llvm-readobj
          llvm-readelf
          llvm-rtdyld
          llvm-size
          llvm-split
          llvm-strings
          llvm-symbolizer
          llvm-tblgen
          llvm-xray
          not
          obj2yaml
          opt
          sancov
          sanstats
          verify-uselistorder
          yaml-bench
          yaml2obj
        )

if(TARGET llvm-lto)
  set(LLVM_TEST_DEPENDS ${LLVM_TEST_DEPENDS} llvm-lto)
endif()

# If Intel JIT events are supported, depend on a tool that tests the listener.
if( LLVM_USE_INTEL_JITEVENTS )
  set(LLVM_TEST_DEPENDS ${LLVM_TEST_DEPENDS} llvm-jitlistener)
endif( LLVM_USE_INTEL_JITEVENTS )

if(TARGET LLVMgold)
  set(LLVM_TEST_DEPENDS ${LLVM_TEST_DEPENDS} LLVMgold)
endif()

if(TARGET llvm-go)
  set(LLVM_TEST_DEPENDS ${LLVM_TEST_DEPENDS} llvm-go)
endif()

if(TARGET LTO)
  set(LLVM_TEST_DEPENDS ${LLVM_TEST_DEPENDS} LTO)
endif()

if(LLVM_BUILD_EXAMPLES)
  list(APPEND LLVM_TEST_DEPENDS
    Kaleidoscope-Ch3
    Kaleidoscope-Ch4
    Kaleidoscope-Ch5
    Kaleidoscope-Ch6
    Kaleidoscope-Ch7
    )
endif()

if(TARGET ocaml_llvm)
  # Clear all non-OCaml cross-target dependencies when building out-of-tree.
  if(LLVM_OCAML_OUT_OF_TREE)
    set(LLVM_TEST_DEPENDS)
  endif()

  set(LLVM_TEST_DEPENDS ${LLVM_TEST_DEPENDS}
          ocaml_llvm
          ocaml_llvm_all_backends
          ocaml_llvm_analysis
          ocaml_llvm_bitreader
          ocaml_llvm_bitwriter
          ocaml_llvm_executionengine
          ocaml_llvm_irreader
          ocaml_llvm_linker
          ocaml_llvm_target
          ocaml_llvm_ipo
          ocaml_llvm_passmgr_builder
          ocaml_llvm_scalar_opts
          ocaml_llvm_transform_utils
          ocaml_llvm_vectorize
        )
endif()

add_custom_target(llvm-test-depends DEPENDS ${LLVM_TEST_DEPENDS})
set_target_properties(llvm-test-depends PROPERTIES FOLDER "Tests")

add_lit_testsuite(check-llvm "Running the LLVM regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LLVM_TEST_DEPENDS}
  )
set_target_properties(check-llvm PROPERTIES FOLDER "Tests")

add_lit_testsuites(LLVM ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${LLVM_TEST_DEPENDS}
  )

# Setup a legacy alias for 'check-llvm'. This will likely change to be an
# alias for 'check-all' at some point in the future.
add_custom_target(check)
add_dependencies(check check-llvm)
set_target_properties(check PROPERTIES FOLDER "Tests")

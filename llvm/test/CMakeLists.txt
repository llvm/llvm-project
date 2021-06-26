llvm_canonicalize_cmake_booleans(
  BUILD_SHARED_LIBS
  LLVM_HAVE_LIBXAR
  HAVE_OCAMLOPT
  HAVE_OCAML_OUNIT
  LLVM_ENABLE_DIA_SDK
  LLVM_ENABLE_FFI
  LLVM_ENABLE_THREADS
  LLVM_ENABLE_ZLIB
  LLVM_ENABLE_LIBXML2
  LLVM_INCLUDE_GO_TESTS
  LLVM_LINK_LLVM_DYLIB
  LLVM_TOOL_LTO_BUILD
  LLVM_USE_INTEL_JITEVENTS
  LLVM_BUILD_EXAMPLES
  LLVM_ENABLE_PLUGINS
  LLVM_BYE_LINK_INTO_TOOLS
  LLVM_HAVE_TF_AOT
  LLVM_HAVE_TF_API
  LLVM_ENABLE_EXPENSIVE_CHECKS
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "CMAKE_OSX_SYSROOT"
  "LLVM_SOURCE_DIR"
  "LLVM_BINARY_DIR"
  "LLVM_TOOLS_DIR"
  "LLVM_LIBS_DIR"
  "SHLIBDIR"
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  PATHS
  "LLVM_SOURCE_DIR"
  "LLVM_BINARY_DIR"
  "LLVM_TOOLS_DIR"
  "SHLIBDIR"
  )

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
          llvm-addr2line
          llvm-ar
          llvm-as
          llvm-bcanalyzer
          llvm-bitcode-strip
          llvm-c-test
          llvm-cat
          llvm-cfi-verify
          llvm-config
          llvm-cov
          llvm-cvtres
          llvm-cxxdump
          llvm-cxxfilt
          llvm-cxxmap
          llvm-diff
          llvm-dis
          llvm-dlltool
          dsymutil
          llvm-dwarfdump
          llvm-dwp
          llvm-elfabi
          llvm-exegesis
          llvm-extract
          llvm-gsymutil
          llvm-isel-fuzzer
          llvm-ifs
          llvm-install-name-tool
          llvm-jitlink
          llvm-lib
          llvm-libtool-darwin
          llvm-link
          llvm-lipo
          llvm-locstats
          llvm-lto2
          llvm-mc
          llvm-mca
          llvm-ml
          llvm-modextract
          llvm-mt
          llvm-nm
          llvm-objcopy
          llvm-objdump
          llvm-opt-fuzzer
          llvm-opt-report
          llvm-otool
          llvm-pdbutil
          llvm-profdata
          llvm-profgen
          llvm-ranlib
          llvm-rc
          llvm-readobj
          llvm-readelf
          llvm-reduce
          llvm-rtdyld
          llvm-sim
          llvm-size
          llvm-split
          llvm-strings
          llvm-strip
          llvm-symbolizer
          llvm-tblgen
          llvm-tapi-diff
          llvm-undname
          llvm-windres
          llvm-xray
          not
          obj2yaml
          opt
          sancov
          sanstats
          split-file
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
    LLJITWithThinLTOSummaries
    )
  if(CMAKE_HOST_UNIX)
    list(APPEND LLVM_TEST_DEPENDS
      LLJITWithRemoteDebugging
      )
  endif()
  if (NOT WIN32)
    list(APPEND LLVM_TEST_DEPENDS
      Bye
      )
  endif()
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

if(LLVM_BUILD_TOOLS)
  set(exclude_from_check_all "")
else()
  set(exclude_from_check_all "EXCLUDE_FROM_CHECK_ALL")
endif()

add_lit_testsuite(check-llvm "Running the LLVM regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ${exclude_from_check_all}
  DEPENDS ${LLVM_TEST_DEPENDS}
  )
set_target_properties(check-llvm PROPERTIES FOLDER "Tests")

add_lit_testsuites(LLVM ${CMAKE_CURRENT_SOURCE_DIR}
  ${exclude_from_check_all}
  DEPENDS ${LLVM_TEST_DEPENDS}
  )

# Setup an alias for 'check-all'.
add_custom_target(check)
add_dependencies(check check-all)
set_target_properties(check PROPERTIES FOLDER "Tests")

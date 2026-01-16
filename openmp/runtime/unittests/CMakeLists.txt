add_custom_target(OpenMPUnitTests)
set_target_properties(OpenMPUnitTests PROPERTIES FOLDER "OpenMP/Tests")

if(WIN32 OR STUBS_LIBRARY)
  message(WARNING "OpenMP unittests disabled due to stub library or Windows")
  return()
endif()

function(standalone_add_gtest success_var)
  # LLVM_RUNTIMES_BUILD avoids gtest to link to LLVMSupport, which might be
  # incompable with the platform libomp is compiled for.
  set(LLVM_RUNTIMES_BUILD 1)
  if (EXISTS "${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h")
    set(${success_var} TRUE PARENT_SCOPE)
    add_subdirectory("${LLVM_THIRD_PARTY_DIR}/unittest" "${CMAKE_BINARY_DIR}/third-party/runtimes_gtest")
  else()
    set(${success_var} FALSE PARENT_SCOPE)
    message(WARNING "OpenMP unit tests will be skipped as LLVM third-party unittest directory is not available at: ${LLVM_THIRD_PARTY_DIR}/unittest")
    return()
  endif()
endfunction()

# Make the targets default_gtest and default_gtest_main available.
if (OPENMP_STANDALONE_BUILD)
  standalone_add_gtest(STANDALONE_UNIT_TESTS)
  if(NOT STANDALONE_UNIT_TESTS)
    return()
  endif()
elseif ("openmp" IN_LIST LLVM_ENABLE_PROJECTS)
  # llvm_gtest should already exist
else ()
  # LLVM_ENABLE_RUNTIMES build
  build_gtest()
endif ()

# Build a testing version of libomp that exports all symbols for unit tests.
# This library uses the same compiled objects as libomp, but with all symbols
# exported to allow testing internal functions.

libomp_get_ldflags(LIBOMP_TEST_LDFLAGS FOR_UNITTESTS)
libomp_get_libflags(LIBOMP_TEST_LIBFLAGS)

# Create the testing library from the same objects as libomp.
add_library(omp_testing SHARED $<TARGET_OBJECTS:obj.omp>)
set_property(TARGET omp_testing PROPERTY FOLDER "OpenMP/Libraries")
target_link_libraries(omp_testing PUBLIC default_gtest)
target_link_libraries(omp_testing PUBLIC ${LIBOMP_TEST_LIBFLAGS} ${LIBOMP_DL_LIBS})
set_target_properties(omp_testing PROPERTIES
  PREFIX ""
  SUFFIX ""
  OUTPUT_NAME "libomp_testing${LIBOMP_LIBRARY_SUFFIX}"
  LINK_FLAGS "${LIBOMP_TEST_LDFLAGS}"
  LINKER_LANGUAGE ${LIBOMP_LINKER_LANGUAGE}
  EXCLUDE_FROM_ALL TRUE  # Don't build by default, only when tests need it
)
target_include_directories(omp_testing PUBLIC
  ${LIBOMP_INCLUDE_DIR}
  ${LIBOMP_SRC_DIR}
)

function(add_openmp_unittest test_name)
  add_unittest(OpenMPUnitTests ${test_name} ${ARGN})

  # Link against the testing library which exports all symbols.
  target_link_libraries(${test_name} PRIVATE omp_testing)

  if(TARGET default_gtest)
    target_link_libraries(${test_name} PRIVATE default_gtest_main default_gtest)
  else ()
    target_link_libraries(${test_name} PRIVATE llvm_gtest_main llvm_gtest)
  endif ()
endfunction()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/../test/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/../test/Unit/lit.cfg.py
)

add_openmp_testsuite(
  check-libomp-unit
  "Running libomp unit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS OpenMPUnitTests
)

add_subdirectory(String)

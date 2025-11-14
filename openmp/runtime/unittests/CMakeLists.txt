add_custom_target(OpenMPUnitTests)
set_target_properties(OpenMPUnitTests PROPERTIES FOLDER "OpenMP/Tests")

if (NOT TARGET llvm_gtest)
  message(WARNING "OpenMP unittests disabled due to GTest being unavailable; "
                  "Try LLVM_INSTALL_GTEST=ON for the LLVM build")
  return ()
endif ()

function(add_openmp_unittest test_dirname)
  add_unittest(OpenMPUnitTests ${test_dirname} ${ARGN})

  # Link against the object library created in runtime/src
  target_link_libraries(${test_dirname} PRIVATE
    $<TARGET_OBJECTS:omp_objects>
  )

  target_include_directories(${test_dirname} PRIVATE
    ${LIBOMP_INCLUDE_DIR}
    ${LIBOMP_SRC_DIR}
  )
endfunction()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_openmp_testsuite(
  check-libomp-unit
  "Running libomp unit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS OpenMPUnitTests omp
)

add_subdirectory(src)

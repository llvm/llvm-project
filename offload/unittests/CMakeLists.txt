add_custom_target(OffloadUnitTests)
set_target_properties(OffloadUnitTests PROPERTIES FOLDER "Tests/UnitTests")

function(add_offload_unittest test_dirname)
  set(TARGET_NAME "${test_dirname}.unittests")

  list(TRANSFORM ARGN PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE FILES)

  add_unittest(OffloadUnitTests "${TARGET_NAME}"
    ${CMAKE_CURRENT_SOURCE_DIR}/common/Environment.cpp
    ${FILES})
  add_dependencies(${TARGET_NAME} ${PLUGINS_TEST_COMMON} OffloadUnitTestsDeviceBins)
  target_compile_definitions(${TARGET_NAME} PRIVATE DEVICE_CODE_PATH="${OFFLOAD_TEST_DEVICE_CODE_PATH}")
  target_link_libraries(${TARGET_NAME} PRIVATE ${PLUGINS_TEST_COMMON})
  target_include_directories(${TARGET_NAME} PRIVATE ${PLUGINS_TEST_INCLUDE})
endfunction()

add_subdirectory(OffloadAPI)

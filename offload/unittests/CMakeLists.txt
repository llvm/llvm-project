add_custom_target(OffloadUnitTests)
set_target_properties(OffloadUnitTests PROPERTIES FOLDER "Tests/UnitTests")

if (CMAKE_CROSSCOMPILING)
  # TODO: It is possible that LLVM_GTEST_RUN_UNDER defines an emulator or
  #       ssh remote command invocation; for this case provide an option to
  #       enable unittests.
  message(STATUS "Offload unittests disabled because we are cross-compiling")
  return ()
endif ()

if (NOT TARGET llvm_gtest)
  message(WARNING "Offload unittests disabled due to GTest being unavailable; "
                  "Try LLVM_INSTALL_GTEST=ON for the LLVM build")
  return ()
endif ()

function(add_offload_unittest test_dirname)
  set(target_name "${test_dirname}.unittests")

  list(TRANSFORM ARGN PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE files)

  add_unittest(OffloadUnitTests "${target_name}"
    ${CMAKE_CURRENT_SOURCE_DIR}/common/Environment.cpp
    ${files})
  add_dependencies(${target_name} ${PLUGINS_TEST_COMMON} OffloadUnitTestsDeviceBins)
  target_compile_definitions(${target_name} PRIVATE DEVICE_CODE_PATH="${OFFLOAD_TEST_DEVICE_CODE_PATH}")
  target_link_libraries(${target_name} PRIVATE ${PLUGINS_TEST_COMMON})
  target_include_directories(${target_name} PRIVATE ${PLUGINS_TEST_INCLUDE})
endfunction()

add_subdirectory(OffloadAPI)

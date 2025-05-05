add_custom_target(OffloadUnitTests)
set_target_properties(OffloadUnitTests PROPERTIES FOLDER "Tests/UnitTests")

function(add_offload_unittest test_dirname)
  add_unittest(OffloadUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(OffloadAPI)

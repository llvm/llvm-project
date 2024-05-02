add_custom_target(LibomptUnitTests)
set_target_properties(LibomptUnitTests PROPERTIES FOLDER "Tests/UnitTests")

function(add_libompt_unittest test_dirname)
  add_unittest(LibomptUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Plugins)

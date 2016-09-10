add_custom_target(StreamExecutorUnitTests)
set_target_properties(StreamExecutorUnitTests PROPERTIES FOLDER "streamexecutor tests")

function(add_se_unittest testdir_name)
  add_unittest(StreamExecutorUnitTests ${testdir_name} ${ARGN})
  target_link_libraries(${testdir_name} streamexecutor)
endfunction()

add_subdirectory(CoreTests)

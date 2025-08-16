add_custom_target(OrcRTUnitTests)
set_target_properties(OrcRTUnitTests PROPERTIES FOLDER "orcrt/Tests")

function(add_orc_rt_unittest test_dirname)
  add_unittest(OrcRTUnitTests ${test_dirname} ${ARGN})
endfunction()

add_orc_rt_unittest(CoreTests
  init.cpp
  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

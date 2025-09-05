add_custom_target(OrcRTUnitTests)
set_target_properties(OrcRTUnitTests PROPERTIES FOLDER "orc-rt/Tests")

if (NOT TARGET llvm_gtest)
  message(WARNING "orc-rt unittests disabled due to GTest being unavailable; "
                  "Try LLVM_INSTALL_GTEST=ON for the LLVM build")
  return ()
endif ()

function(add_orc_rt_unittest test_dirname)
  add_unittest(OrcRTUnitTests ${test_dirname} ${ARGN})
endfunction()

add_orc_rt_unittest(CoreTests
  BitmaskEnumTest.cpp
  CommonTestUtils.cpp
  ErrorTest.cpp
  ExecutorAddressTest.cpp
  IntervalMapTest.cpp
  IntervalSetTest.cpp
  MathTest.cpp
  MemoryFlagsTest.cpp
  RTTITest.cpp
  SimplePackedSerializationTest.cpp
  WrapperFunctionBufferTest.cpp
  bind-test.cpp
  bit-test.cpp
  move_only_function-test.cpp
  span-test.cpp
  DISABLE_LLVM_LINK_LLVM_DYLIB
  )
target_link_libraries(CoreTests PRIVATE orc-rt-executor)

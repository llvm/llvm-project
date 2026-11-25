add_unittest(OffloadUnitTests ompt.unittests
  TraceBufferTest.cpp)
target_link_libraries(ompt.unittests PRIVATE omptarget.ompt)

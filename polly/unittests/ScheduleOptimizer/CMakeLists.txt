set(LLVM_LINK_COMPONENTS
  SelectionDAG
  )
add_polly_unittest(ScheduleOptimizerTests
    ScheduleTreeTransformTest.cpp
  )

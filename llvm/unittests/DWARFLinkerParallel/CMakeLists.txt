set(LLVM_LINK_COMPONENTS
  Support
  DWARFLinkerParallel
  )

add_llvm_unittest(DWARFLinkerParallelTests
  DWARFLinkerTest.cpp
  StringPoolTest.cpp
  )

target_link_libraries(DWARFLinkerParallelTests PRIVATE LLVMTestingSupport)

add_dependencies(DWARFLinkerParallelTests intrinsics_gen)

set(LLVM_LINK_COMPONENTS
  DWARFLinker
  DWARFLinkerParallel
  Support
  )

add_llvm_unittest(DWARFLinkerParallelTests
  DWARFLinkerTest.cpp
  StringPoolTest.cpp
  )

target_link_libraries(DWARFLinkerParallelTests PRIVATE LLVMTestingSupport)

add_dependencies(DWARFLinkerParallelTests intrinsics_gen)

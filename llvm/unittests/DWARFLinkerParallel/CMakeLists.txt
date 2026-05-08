set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  DWARFLinker
  DWARFLinkerParallel
  Support
  )

add_llvm_unittest(DWARFLinkerParallelTests
  DWARFLinkerTest.cpp
  StringPoolTest.cpp
  )

target_link_libraries(DWARFLinkerParallelTests PRIVATE
  LLVMTestingSupport
  LLVMDebugInfoDWARF
  )

add_dependencies(DWARFLinkerParallelTests intrinsics_gen)

set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  DWARFLinker
  DWARFLinkerParallel
  Support
  )

add_llvm_unittest(DWARFLinkerParallelTests
  DIEInfoPlacementTest.cpp
  DWARFLinkerTest.cpp
  StringPoolTest.cpp
  TypePoolTest.cpp
  )

target_link_libraries(DWARFLinkerParallelTests PRIVATE
  LLVMTestingSupport
  )

# DIEInfoPlacementTest.cpp and TypePoolTest.cpp exercise classes defined in the
# library's private headers.
target_include_directories(DWARFLinkerParallelTests PRIVATE
  ${LLVM_MAIN_SRC_DIR}/lib/DWARFLinker/Parallel
  )

add_dependencies(DWARFLinkerParallelTests intrinsics_gen)

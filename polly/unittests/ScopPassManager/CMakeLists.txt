llvm_map_components_to_libnames(llvm_libs Passes Core Analysis)
add_polly_unittest(ScopPassManagerTests
  PassManagerTest.cpp
  )
target_link_libraries(ScopPassManagerTests PRIVATE ${llvm_libs})

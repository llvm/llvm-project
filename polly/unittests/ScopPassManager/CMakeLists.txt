add_polly_unittest(ScopPassManagerTests
  PassManagerTest.cpp
  )
if (NOT LLVM_LINK_LLVM_DYLIB)
  llvm_map_components_to_libnames(llvm_libs Passes Core Analysis)
  target_link_libraries(ScopPassManagerTests PRIVATE ${llvm_libs})
endif()

add_mlir_library(MLIRTestRegionBranchOpInterface
  TestRegionBranchOpInterface.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRTestRegionBranchOpInterface PUBLIC
  MLIRControlFlowInterfaces
  MLIRPass
  )

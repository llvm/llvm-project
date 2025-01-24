add_mlir_tool(mlir-irdl-to-cpp
    mlir-irdl-to-cpp.cpp
  )
mlir_target_link_libraries(mlir-irdl-to-cpp
  PRIVATE
  MLIRTargetIRDLToCpp
  )

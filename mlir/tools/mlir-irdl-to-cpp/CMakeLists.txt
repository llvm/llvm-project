add_llvm_executable(mlir-irdl-to-cpp
    mlir-irdl-to-cpp.cpp
  )
mlir_target_link_libraries(mlir-irdl-to-cpp
  PRIVATE
  MLIRTargetIRDLToCpp
  )

setup_host_tool(mlir-irdl-to-cpp MLIR_IRDL_TO_CPP MLIR_IRDL_TO_CPP_EXE MLIR_IRDL_TO_CPP_TARGET)

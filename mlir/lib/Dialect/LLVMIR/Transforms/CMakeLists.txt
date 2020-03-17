add_mlir_dialect_library(MLIRLLVMIRTransforms
  LegalizeForExport.cpp
  )

target_link_libraries(MLIRLLVMIRTransforms
  PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  )

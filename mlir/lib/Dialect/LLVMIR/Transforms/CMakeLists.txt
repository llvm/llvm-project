add_mlir_dialect_library(MLIRLLVMIRTransforms
  LegalizeForExport.cpp

  DEPENDS
  MLIRLLVMPassIncGen
  )

target_link_libraries(MLIRLLVMIRTransforms
  PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  )

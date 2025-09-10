add_mlir_library(MLIRPDLLCodeGen
  CPPGen.cpp
  MLIRGen.cpp

  LINK_LIBS PUBLIC
  MLIRParser
  MLIRPDLLAST
  MLIRPDLDialect
  MLIRSupport
  )
set_target_properties(obj.MLIRPDLLCodeGen PROPERTIES UNITY_BUILD_BATCH_SIZE 1)

add_mlir_unittest(MLIRPDLToPDLInterpTests
  RootOrderingTest.cpp
)
mlir_target_link_libraries(MLIRPDLToPDLInterpTests
  PRIVATE
  MLIRArithDialect
  MLIRPDLToPDLInterp
)

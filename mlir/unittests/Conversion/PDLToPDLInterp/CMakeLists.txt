add_mlir_unittest(MLIRPDLToPDLInterpTests
  RootOrderingTest.cpp
)
target_link_libraries(MLIRPDLToPDLInterpTests
  PRIVATE
  MLIRArithDialect
  MLIRPDLToPDLInterp
)

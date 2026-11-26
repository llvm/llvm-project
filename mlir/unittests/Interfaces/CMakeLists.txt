add_mlir_unittest(MLIRInterfacesTests
  ControlFlowInterfacesTest.cpp
  DataLayoutInterfacesTest.cpp
  InferIntRangeInterfaceTest.cpp
  MemorySlotUtilsTest.cpp
  SideEffectInterfacesTest.cpp
  InferTypeOpInterfaceTest.cpp

  DEPENDS
  MLIRTestInterfaceIncGen
)

mlir_target_link_libraries(MLIRInterfacesTests
  PRIVATE
  MLIRAnalysisBase
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRMemorySlotUtils
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
)
target_link_libraries(MLIRInterfacesTests PRIVATE MLIRTestDialect)

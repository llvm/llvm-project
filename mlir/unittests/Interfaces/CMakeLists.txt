add_mlir_unittest(MLIRInterfacesTests
  ControlFlowInterfacesTest.cpp
  DataLayoutInterfacesTest.cpp
  InferIntRangeInterfaceTest.cpp
  MemorySlotUtilsTest.cpp
  SideEffectInterfacesTest.cpp
  InferTypeOpInterfaceTest.cpp
  ValueBoundsOpInterfaceTest.cpp

  DEPENDS
  MLIRTestInterfaceIncGen
)

mlir_target_link_libraries(MLIRInterfacesTests
  PRIVATE
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
  MLIRSideEffectInterfaces
  MLIRValueBoundsOpInterface
)
target_link_libraries(MLIRInterfacesTests PRIVATE MLIRTestDialect)

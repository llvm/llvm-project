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
target_include_directories(MLIRInterfacesTests PRIVATE "${MLIR_BINARY_DIR}/test/lib/Dialect/Test")

mlir_target_link_libraries(MLIRInterfacesTests
  PRIVATE
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRMemorySlotUtils
  MLIRParser
  MLIRSideEffectInterfaces
)
target_link_libraries(MLIRInterfacesTests PRIVATE MLIRTestDialect)

add_mlir_unittest(MLIRInterfacesTests
  ControlFlowInterfacesTest.cpp
  DataLayoutInterfacesTest.cpp
  InferIntRangeInterfaceTest.cpp
  MemorySlotUtilsTest.cpp
  InferTypeOpInterfaceTest.cpp

  PARTIAL_SOURCES_INTENDED

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
)
target_link_libraries(MLIRInterfacesTests PRIVATE MLIRTestDialect)

add_mlir_unittest(MLIRSideEffectInterfacesTests
  SideEffectInterfacesTest.cpp
)
mlir_target_link_libraries(MLIRSideEffectInterfacesTests
  PRIVATE
  MLIRAnalysisBase
  MLIRAnalysisManager
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRParser
  MLIRSideEffectInterfaces
)

add_mlir_unittest(MLIRInterfacesTests
  ControlFlowInterfacesTest.cpp
  DataLayoutInterfacesTest.cpp
  InferIntRangeInterfaceTest.cpp
  InferTypeOpInterfaceTest.cpp
)

target_link_libraries(MLIRInterfacesTests
  PRIVATE
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDLTI
  MLIRFunc
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRParser
)

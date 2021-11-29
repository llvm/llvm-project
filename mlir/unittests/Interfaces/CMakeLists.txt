add_mlir_unittest(MLIRInterfacesTests
  ControlFlowInterfacesTest.cpp
  DataLayoutInterfacesTest.cpp
  InferTypeOpInterfaceTest.cpp
)

target_link_libraries(MLIRInterfacesTests
  PRIVATE
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDLTI
  MLIRInferTypeOpInterface
  MLIRParser
  MLIRStandard
)

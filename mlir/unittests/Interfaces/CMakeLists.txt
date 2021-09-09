add_mlir_unittest(MLIRInterfacesTests
  DataLayoutInterfacesTest.cpp
  InferTypeOpInterfaceTest.cpp
)

target_link_libraries(MLIRInterfacesTests
  PRIVATE
  MLIRDataLayoutInterfaces
  MLIRDLTI
  MLIRInferTypeOpInterface
  MLIRParser
  MLIRStandard
)

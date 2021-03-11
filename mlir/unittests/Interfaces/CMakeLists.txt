add_mlir_unittest(MLIRInterfacesTests
  DataLayoutInterfacesTest.cpp
)

target_link_libraries(MLIRInterfacesTests
  PRIVATE
  MLIRDataLayoutInterfaces
  MLIRDLTI
  MLIRParser
  LLVMSupport
)

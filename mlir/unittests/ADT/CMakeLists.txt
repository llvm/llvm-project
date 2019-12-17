add_mlir_unittest(MLIRADTTests
  TypeSwitchTest.cpp
)

target_link_libraries(MLIRADTTests PRIVATE MLIRSupport LLVMSupport)

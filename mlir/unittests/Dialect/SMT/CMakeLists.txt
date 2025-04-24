add_mlir_unittest(MLIRSMTTests
  AttributeTest.cpp
  QuantifierTest.cpp
  TypeTest.cpp
)

mlir_target_link_libraries(MLIRSMTTests
  PRIVATE
  MLIRSMT
)

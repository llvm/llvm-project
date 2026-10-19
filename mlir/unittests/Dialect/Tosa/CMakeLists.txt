add_mlir_unittest(MLIRTosaTests
  ShapeUtilsTest.cpp
)

mlir_target_link_libraries(MLIRTosaTests
  PRIVATE
  MLIRIR
  MLIRInferTypeOpInterface
)

add_mlir_unittest(MLIRQuantTests
  QuantizationUtilsTest.cpp
)
target_link_libraries(MLIRQuantTests
  PRIVATE
  MLIRQuant
  MLIRQuantUtils
  )

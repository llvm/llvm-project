add_mlir_unittest(MLIRMemRefTests
  InferShapeTest.cpp
)
mlir_target_link_libraries(MLIRMemRefTests
  PRIVATE
  MLIRMemRefDialect
  )

add_mlir_unittest(MLIRMemRefTests
  InferShapeTest.cpp
)
target_link_libraries(MLIRMemRefTests
  PRIVATE
  MLIRMemRefDialect
  )

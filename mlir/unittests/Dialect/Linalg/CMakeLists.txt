add_mlir_unittest(MLIRLinalgTests
  LinalgInterfacesTest.cpp
)
target_link_libraries(MLIRLinalgTests
  PRIVATE
  MLIRLinalgDialect
  )


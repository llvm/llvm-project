add_mlir_unittest(MLIRABITests
  ABIRewriteContextTest.cpp
  ABITypeMapperTest.cpp
)

mlir_target_link_libraries(MLIRABITests
  PRIVATE
  MLIRABI
  MLIRDataLayoutInterfaces
  MLIRDLTIDialect
  MLIRIR
)

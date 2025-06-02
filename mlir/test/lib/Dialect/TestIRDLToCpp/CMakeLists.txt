add_irdl_to_cpp_target(TestIRDLToCppGen test_irdl_to_cpp.irdl.mlir)

add_mlir_library(MLIRTestIRDLToCppDialect
  TestIRDLToCppDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  TestIRDLToCppGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
)

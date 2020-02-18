add_mlir_unittest(MLIRSPIRVTests
  DeserializationTest.cpp
  SerializationTest.cpp
)
target_link_libraries(MLIRSPIRVTests
  PRIVATE
  MLIRIR
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRStandardOps
  MLIRSupport)

whole_archive_link(MLIRSPIRVTests MLIRSPIRV)


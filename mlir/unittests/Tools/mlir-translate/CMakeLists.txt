add_mlir_unittest(MLIRTranslateMainTests
  MlirTranslateMainTest.cpp
)

target_link_libraries(MLIRTranslateMainTests
  PRIVATE
  MLIRTranslateLib
)

add_mlir_dialect_library(MLIRTosaTestPasses
  TosaTestPasses.cpp
  TestAvailability.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa/Transforms

  DEPENDS
  MLIRTosaPassIncGen
  )
mlir_target_link_libraries(MLIRTosaTestPasses PUBLIC
  MLIRFuncDialect
  MLIRPass
  MLIRTosaDialect
  MLIRTransformUtils
  )

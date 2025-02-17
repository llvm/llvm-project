add_mlir_library(MLIRTestOneToNTypeConversionPass
  TestOneToNTypeConversionPass.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRTestDialect
 )
mlir_target_link_libraries(MLIRTestOneToNTypeConversionPass PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransformUtils
 )

target_include_directories(MLIRTestOneToNTypeConversionPass
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )

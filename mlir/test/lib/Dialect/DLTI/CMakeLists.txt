# Exclude tests from libMLIR.so
add_mlir_library(MLIRDLTITestPasses
  TestDataLayoutQuery.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRTestDialect
  )
mlir_target_link_libraries(MLIRDLTITestPasses PUBLIC
  MLIRDLTIDialect
  MLIRPass
  )

target_include_directories(MLIRDLTITestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Test
  )

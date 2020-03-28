add_llvm_library(MLIRTestIR
  TestFunc.cpp
  TestMatchers.cpp
  TestSideEffects.cpp
  TestSymbolUses.cpp

  ADDITIONAL_HEADER_DIRS
  )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)

target_link_libraries(MLIRTestIR
  PUBLIC
  MLIRPass
  MLIRTestDialect
  )

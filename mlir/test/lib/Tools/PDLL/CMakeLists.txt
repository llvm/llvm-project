add_mlir_pdll_library(MLIRTestPDLLPatternsIncGen
  TestPDLL.pdll
  TestPDLLPatterns.h.inc

  EXTRA_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )

# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestPDLL
  TestPDLL.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/PDLL

  DEPENDS
  MLIRTestPDLLPatternsIncGen

  LINK_LIBS PUBLIC
  MLIRTestDialect
  )
mlir_target_link_libraries(MLIRTestPDLL PUBLIC
  MLIRCastInterfaces
  MLIRIR
  MLIRPass
  MLIRPDLInterpDialect
  MLIRPDLDialect
  MLIRSupport
  MLIRTransformUtils
  )

target_include_directories(MLIRTestPDLL
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )

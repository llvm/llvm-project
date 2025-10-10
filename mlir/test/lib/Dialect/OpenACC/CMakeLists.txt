add_mlir_library(MLIROpenACCTestPasses
  TestOpenACC.cpp
  TestPointerLikeTypeInterface.cpp
  
  EXCLUDE_FROM_LIBMLIR
)
mlir_target_link_libraries(MLIROpenACCTestPasses PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIROpenACCDialect
  MLIRPass
  MLIRSupport
)


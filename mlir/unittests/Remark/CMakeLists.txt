add_mlir_unittest(MLIRRemarkTests
  LLVMRemarkImportTest.cpp
)
mlir_target_link_libraries(MLIRRemarkTests
  PRIVATE
  MLIRIR
  MLIRParser
)
target_link_libraries(MLIRRemarkTests PRIVATE MLIRRemarkStreamer)

add_mlir_unittest(MLIRLspServerSupportTests
  Transport.cpp
)
target_link_libraries(MLIRLspServerSupportTests
  PRIVATE
  MLIRLspServerSupportLib)

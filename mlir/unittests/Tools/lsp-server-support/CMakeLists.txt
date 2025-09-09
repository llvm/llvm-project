add_mlir_unittest(MLIRLspServerSupportTests
  Protocol.cpp
  Transport.cpp
)
mlir_target_link_libraries(MLIRLspServerSupportTests
  PRIVATE
  MLIRLspServerSupportLib)

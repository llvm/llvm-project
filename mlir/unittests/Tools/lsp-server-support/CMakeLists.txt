add_mlir_unittest(MLIRLspServerSupportTests
  Protocol.cpp
  Transport.cpp
)
target_link_libraries(MLIRLspServerSupportTests
  PRIVATE
  MLIRLspServerSupportLib)

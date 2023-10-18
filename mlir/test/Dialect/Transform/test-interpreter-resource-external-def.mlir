// RUN: cd %p; mlir-opt %s

module attributes { test.blob_ref = #transform.rref<library @foo> : tensor<*xi1>} {
  module attributes { test.blob_ref = #transform.rref<banana @bar> : tensor<*xi1>} {}
  module {}
  module {}
}

{-#
  dialect_resources: {
    transform: {
      library: "test-interpreter-external-source.mlir",
      banana: "test-interpreter-external-source.mlir"
    }
  }
#-}

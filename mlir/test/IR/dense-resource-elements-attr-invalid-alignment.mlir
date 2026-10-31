// RUN: mlir-opt -allow-unregistered-dialect %s -verify-diagnostics

"test.user_op"() {attr = dense_resource<double_data> : tensor<2xf64>} : () -> ()

{-#
  dialect_resources: {
    builtin: {
      // expected-error @+1 {{expected hex string blob for key 'double_data' to encode alignment in first 4 bytes, but got non-power-of-2 value: 0}}
      double_data: "0x000000000000F03F0000000000000040"
    }
  }
#-}

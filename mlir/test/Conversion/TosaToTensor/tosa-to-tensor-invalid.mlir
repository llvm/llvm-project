// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-tensor))" %s -verify-diagnostics

// CHECK-LABEL:  @slice_resultType_unranked
func.func @slice_resultType_unranked(%arg0: tensor<?xf32>) -> (tensor<*xf32>) {
  // expected-error@+1 {{failed to legalize operation 'tosa.slice'}}
  %0 = "tosa.slice"(%arg0) {start = array<i64: 2>, size = array<i64: 0>} : (tensor<?xf32>)  -> (tensor<*xf32>)
  return %0 : tensor<*xf32>
}

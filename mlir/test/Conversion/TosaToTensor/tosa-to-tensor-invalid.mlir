// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-tensor))" %s -verify-diagnostics

// CHECK-LABEL:  @slice_resultType_unranked
func.func @slice_resultType_unranked(%arg0: tensor<?xf32>) -> (tensor<*xf32>) {
  %0 = tosa.const_shape  {value = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = tosa.const_shape  {value = dense<0> : tensor<1xindex>} : () -> !tosa.shape<1>
  // expected-error@+1 {{failed to legalize operation 'tosa.slice'}}
  %2 = tosa.slice %arg0, %0, %1 : (tensor<?xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}

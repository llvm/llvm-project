// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-tensor))" %s -verify-diagnostics

// CHECK-LABEL:  @slice_resultType_unranked
func.func @slice_resultType_unranked(%arg0: tensor<?xf32>) -> (tensor<*xf32>) {
  %size = tosa.const_shape values(dense<2> : tensor<1xindex>) : () -> !tosa.shape<1>
  %start = tosa.const_shape values(dense<0> : tensor<1xindex>) : () -> !tosa.shape<1>
  // expected-error@+1 {{failed to legalize operation 'tosa.slice'}}
  %2 = tosa.slice %arg0, %start, %size : (tensor<?xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}

// -----

func.func @pad_unknown_padding(%arg0: tensor<1x2xf32>) -> tensor<?x9xf32> {
  %padding = tosa.const_shape values(dense<[-1, 2, 3, 4]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %pad_const = "tosa.const"() <{values = dense<3.14> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error@+1 {{failed to legalize operation 'tosa.pad'}}
  %0 = "tosa.pad"(%arg0, %padding, %pad_const) : (tensor<1x2xf32>, !tosa.shape<4>, tensor<1xf32>) -> tensor<?x9xf32>
  return %0 : tensor<?x9xf32>
}

// RUN: mlir-opt --split-input-file --tosa-to-scf %s -verify-diagnostics -o -

// CHECK-LABEL: @scatter_unranked
func.func @scatter_unranked(%v: tensor<*xi32>, %idx: tensor<*xi32>, %inp: tensor<*xi32>) -> tensor<*xi32> {
  // expected-error @+1 {{failed to legalize operation 'tosa.scatter' that was explicitly marked illegal}}
  %0 = tosa.scatter %v, %idx, %inp
      : (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

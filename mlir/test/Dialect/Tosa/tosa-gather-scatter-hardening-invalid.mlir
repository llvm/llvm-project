// RUN: mlir-opt %s -tosa-gather-scatter-hardening -verify-diagnostics

// A correctly hardened gather must not hide a scatter whose indexed dimension
// is dynamic, even when they share the same bounded indices.
func.func @dynamic_indexed_dimension(
    %values: tensor<1x21x1xi8>, %dynamic_values: tensor<1x?x1xi8>,
    %indices: tensor<1x2xi32>, %updates: tensor<1x2x1xi8>)
    -> (tensor<1x2x1xi8>, tensor<1x?x1xi8>) {
  %zero = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %upper = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %0 = tosa.maximum %indices, %zero : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
  %1 = tosa.minimum %0, %upper : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
  %gather = tosa.gather %values, %1 : (tensor<1x21x1xi8>, tensor<1x2xi32>) -> tensor<1x2x1xi8>
  // expected-error@+1 {{'tosa.scatter' op requires a statically known indexed dimension for gather/scatter hardening}}
  %scatter = tosa.scatter %dynamic_values, %1, %updates : (tensor<1x?x1xi8>, tensor<1x2xi32>, tensor<1x2x1xi8>) -> tensor<1x?x1xi8>
  return %gather, %scatter : tensor<1x2x1xi8>, tensor<1x?x1xi8>
}

// RUN: not mlir-opt %s -tosa-gather-scatter-hardening 2>&1 | FileCheck %s

func.func @dynamic_indexed_dimension(%arg0: tensor<3x?x5xi8>,
                                     %arg1: tensor<3x6xi32>)
    -> tensor<3x6x5xi8> {
  // CHECK: error: 'tosa.gather' op requires a statically known indexed dimension for gather/scatter hardening
  %0 = tosa.gather %arg0, %arg1 : (tensor<3x?x5xi8>, tensor<3x6xi32>) -> tensor<3x6x5xi8>
  return %0 : tensor<3x6x5xi8>
}

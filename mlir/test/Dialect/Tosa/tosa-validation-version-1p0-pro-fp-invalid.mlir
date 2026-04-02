// RUN: mlir-opt %s -verify-diagnostics -tosa-attach-target="specification_version=1.0 profiles=pro_fp" -tosa-validate="strict-op-spec-alignment"

func.func @test_gather_i8_i32(%input: tensor<13x27x3xi8>, %indices: tensor<13x26xi32>) -> tensor<13x26x3xi8> {
  // expected-error@+1 {{'tosa.gather' op illegal: requires any of [pro_int] profiles/extensions OR requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %gather = tosa.gather %input, %indices : (tensor<13x27x3xi8>, tensor<13x26xi32>) -> tensor<13x26x3xi8>
  return %gather : tensor<13x26x3xi8>
}

// -----

func.func @test_scatter_i8_i32(%input: tensor<13x27x3xi8>, %indices: tensor<13x26xi32>, %updates: tensor<13x26x3xi8>) -> tensor<13x27x3xi8> {
  // expected-error@+1 {{'tosa.scatter' op illegal: requires any of [pro_int] profiles/extensions OR requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %scatter = tosa.scatter %input, %indices, %updates : (tensor<13x27x3xi8>, tensor<13x26xi32>, tensor<13x26x3xi8>) -> tensor<13x27x3xi8>
  return %scatter : tensor<13x27x3xi8>
}

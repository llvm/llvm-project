// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' -verify-diagnostics

// -----

func.func @test_invalid_reify_dim(%size: index) -> (index) {
    %zero = arith.constant 0 : index
    %tensor_val = tensor.empty(%size) : tensor<?xf32>

    // expected-error@+1 {{'test.reify_bound' op invalid dim for shaped type}}
    %dim = "test.reify_bound"(%tensor_val) {dim = 1 : i64} : (tensor<?xf32>) -> index

    return %dim: index
}

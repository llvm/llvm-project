// RUN: mlir-opt -split-input-file %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' -verify-diagnostics

func.func @test_invalid_reify_dim(%size: index) -> (index) {
    %zero = arith.constant 0 : index
    %tensor_val = tensor.empty(%size) : tensor<?xf32>

    // expected-error@+1 {{'test.reify_bound' op invalid dim for shaped type}}
    %dim = "test.reify_bound"(%tensor_val) {dim = 1 : i64} : (tensor<?xf32>) -> index

    return %dim: index
}

// -----

func.func @test_invalid_reify_negative_dim(%size: index) -> (index) {
    %zero = arith.constant 0 : index
    %tensor_val = tensor.empty(%size) : tensor<?xf32>

    // expected-error@+1 {{'test.reify_bound' op dim must be non-negative}}
    %dim = "test.reify_bound"(%tensor_val) {dim = -1 : i64} : (tensor<?xf32>) -> index

    return %dim: index
}

// -----

func.func @test_invalid_reify_int_value(%size: index) -> (index) {
    %zero = arith.constant 0 : index
    %int_val = arith.constant 1 : index

    // expected-error@+1 {{'test.reify_bound' op unexpected 'dim' attribute for index variable}}
    %dim = "test.reify_bound"(%int_val) {dim = 1 : i64} : (index) -> index

    return %dim: index
}

// -----

func.func @test_invalid_reify_without_dim(%size: index) -> (index) {
    %zero = arith.constant 0 : index
    %tensor_val = tensor.empty(%size) : tensor<?xf32>

    // expected-error@+1 {{'test.reify_bound' op expected 'dim' attribute for shaped type variable}}
    %dim = "test.reify_bound"(%tensor_val) : (tensor<?xf32>) -> index

    return %dim: index
}
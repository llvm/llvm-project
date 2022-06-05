// RUN: mlir-opt %s --allow-unregistered-dialect | mlir-opt --allow-unregistered-dialect | FileCheck %s
// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | mlir-opt --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: ml_program.func private @extern_func
ml_program.func private @extern_func(i32) -> i32

// CHECK-LABEL: ml_program.func @defined_func
ml_program.func @defined_func(%arg0 : i32) -> i32 {
  ml_program.return %arg0 : i32
}

// CHECK-LABEL: ml_program.subgraph private @extern_subgraph
ml_program.subgraph private @extern_subgraph(i32) -> i32

// CHECK-LABEL: ml_program.subgraph @compute_subgraph
ml_program.subgraph @compute_subgraph(%arg0 : i32) -> i32 {
  %token = ml_program.token
  %1 = "unregistered.dummy"(%0, %token) : (i32, !ml_program.token) -> i32
  %0 = "unregistered.dummy"(%arg0) : (i32) -> i32
  ml_program.output %0 : i32
}

// CHECK: ml_program.global private @global_same_type(dense<4> : tensor<4xi32>) : tensor<4xi32>
ml_program.global private @global_same_type(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK: ml_program.global private mutable @global_mutable_undef : tensor<?xi32>
ml_program.global private mutable @global_mutable_undef : tensor<?xi32>

// CHECK: ml_program.global private mutable @global_extern(#extern) : tensor<?xi32>
ml_program.global private mutable @global_extern(#ml_program.extern : tensor<4xi32>) : tensor<?xi32>

// CHECK-LABEL: @global_load_const
ml_program.func @global_load_const() -> tensor<4xi32> {
  %0 = ml_program.global_load_const @global_same_type : tensor<4xi32>
  ml_program.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @global_load_store
ml_program.func @global_load_store() {
  %0 = ml_program.global_load @global_mutable_undef : tensor<?xi32>
  ml_program.global_store @global_mutable_undef = %0 : tensor<?xi32>
  ml_program.return
}

// CHECK-LABEL: @global_load_store_tokens
ml_program.subgraph @global_load_store_tokens() -> (tensor<?xi32>, !ml_program.token) {
  %token1 = ml_program.token
  %0, %token2 = ml_program.global_load @global_mutable_undef
      ordering(() -> !ml_program.token) : tensor<?xi32>
  %token3 = ml_program.global_store @global_mutable_undef = %0
      ordering(%token1, %token2 -> !ml_program.token) : tensor<?xi32>
  ml_program.global_store @global_mutable_undef = %0
      ordering(%token3) : tensor<?xi32>

  ml_program.output %0, %token3 : tensor<?xi32>, !ml_program.token
}

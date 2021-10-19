// RUN: mlir-opt %s -async-runtime-ref-counting-opt | FileCheck %s

func private @consume_token(%arg0: !async.token)

// CHECK-LABEL: @cancellable_operations_0
func @cancellable_operations_0(%arg0: !async.token) {
  // CHECK-NOT: async.runtime.add_ref
  // CHECK-NOT: async.runtime.drop_ref
  async.runtime.add_ref %arg0 {count = 1 : i64} : !async.token
  async.runtime.drop_ref %arg0 {count = 1 : i64} : !async.token
  // CHECK: return
  return
}

// CHECK-LABEL: @cancellable_operations_1
func @cancellable_operations_1(%arg0: !async.token) {
  // CHECK-NOT: async.runtime.add_ref
  async.runtime.add_ref %arg0 {count = 1 : i64} : !async.token
  // CHECK: call @consume_toke
  call @consume_token(%arg0): (!async.token) -> ()
  // CHECK-NOT: async.runtime.drop_ref
  async.runtime.drop_ref %arg0 {count = 1 : i64} : !async.token
  // CHECK: return
  return
}

// CHECK-LABEL: @cancellable_operations_2
func @cancellable_operations_2(%arg0: !async.token) {
  // CHECK: async.runtime.await
  // CHECK-NEXT: async.runtime.await
  // CHECK-NEXT: async.runtime.await
  // CHECK-NEXT: return
  async.runtime.add_ref %arg0 {count = 1 : i64} : !async.token
  async.runtime.await %arg0 : !async.token
  async.runtime.drop_ref %arg0 {count = 1 : i64} : !async.token
  async.runtime.await %arg0 : !async.token
  async.runtime.add_ref %arg0 {count = 1 : i64} : !async.token
  async.runtime.await %arg0 : !async.token
  async.runtime.drop_ref %arg0 {count = 1 : i64} : !async.token
  return
}

// CHECK-LABEL: @cancellable_operations_3
func @cancellable_operations_3(%arg0: !async.token) {
  // CHECK-NOT: add_ref
  async.runtime.add_ref %arg0 {count = 1 : i64} : !async.token
  // CHECK: call @consume_toke
  call @consume_token(%arg0): (!async.token) -> ()
  // CHECK-NOT: async.runtime.drop_ref
  async.runtime.drop_ref %arg0 {count = 1 : i64} : !async.token
  // CHECK: async.runtime.await
  async.runtime.await %arg0 : !async.token
  // CHECK: return
  return
}

// CHECK-LABEL: @not_cancellable_operations_0
func @not_cancellable_operations_0(%arg0: !async.token) {
  // CHECK: add_ref
  async.runtime.add_ref %arg0 {count = 1 : i64} : !async.token
  // CHECK: call @consume_toke
  call @consume_token(%arg0): (!async.token) -> ()
  // CHECK: async.runtime.await
  async.runtime.await %arg0 : !async.token
  // CHECK: async.runtime.drop_ref
  async.runtime.drop_ref %arg0 {count = 1 : i64} : !async.token
  // CHECK: return
  return
}

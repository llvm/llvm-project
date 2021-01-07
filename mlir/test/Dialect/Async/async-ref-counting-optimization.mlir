// RUN: mlir-opt %s -async-ref-counting-optimization | FileCheck %s

// CHECK-LABEL: @cancellable_operations_0
func @cancellable_operations_0(%arg0: !async.token) {
  // CHECK-NOT: async.add_ref
  // CHECK-NOT: async.drop_ref
  async.add_ref %arg0 {count = 1 : i32} : !async.token
  async.drop_ref %arg0 {count = 1 : i32} : !async.token
  // CHECK: return
  return
}

// CHECK-LABEL: @cancellable_operations_1
func @cancellable_operations_1(%arg0: !async.token) {
  // CHECK-NOT: async.add_ref
  // CHECK: async.execute
  async.add_ref %arg0 {count = 1 : i32} : !async.token
  async.execute [%arg0] {
    // CHECK: async.drop_ref
    async.drop_ref %arg0 {count = 1 : i32} : !async.token
    // CHECK-NEXT: async.yield
    async.yield
  }
  // CHECK-NOT: async.drop_ref
  async.drop_ref %arg0 {count = 1 : i32} : !async.token
  // CHECK: return
  return
}

// CHECK-LABEL: @cancellable_operations_2
func @cancellable_operations_2(%arg0: !async.token) {
  // CHECK: async.await
  // CHECK-NEXT: async.await
  // CHECK-NEXT: async.await
  // CHECK-NEXT: return
  async.add_ref %arg0 {count = 1 : i32} : !async.token
  async.await %arg0 : !async.token
  async.drop_ref %arg0 {count = 1 : i32} : !async.token
  async.await %arg0 : !async.token
  async.add_ref %arg0 {count = 1 : i32} : !async.token
  async.await %arg0 : !async.token
  async.drop_ref %arg0 {count = 1 : i32} : !async.token
  return
}

// CHECK-LABEL: @cancellable_operations_3
func @cancellable_operations_3(%arg0: !async.token) {
  // CHECK-NOT: add_ref
  async.add_ref %arg0 {count = 1 : i32} : !async.token
  %token = async.execute {
    async.await %arg0 : !async.token
    // CHECK: async.drop_ref
    async.drop_ref %arg0 {count = 1 : i32} : !async.token
    async.yield
  }
  // CHECK-NOT: async.drop_ref
  async.drop_ref %arg0 {count = 1 : i32} : !async.token
  // CHECK: async.await
  async.await %arg0 : !async.token
  // CHECK: return
  return
}

// CHECK-LABEL: @not_cancellable_operations_0
func @not_cancellable_operations_0(%arg0: !async.token, %arg1: i1) {
  // It is unsafe to cancel `add_ref` / `drop_ref` pair because it is possible
  // that the body of the `async.execute` operation will run before the await
  // operation in the function body, and will destroy the `%arg0` token.
  // CHECK: add_ref
  async.add_ref %arg0 {count = 1 : i32} : !async.token
  %token = async.execute {
    // CHECK: async.await
    async.await %arg0 : !async.token
    // CHECK: async.drop_ref
    async.drop_ref %arg0 {count = 1 : i32} : !async.token
    // CHECK: async.yield
    async.yield
  }
  // CHECK: async.await
  async.await %arg0 : !async.token
  // CHECK: drop_ref
  async.drop_ref %arg0 {count = 1 : i32} : !async.token
  // CHECK: return
  return
}

// CHECK-LABEL: @not_cancellable_operations_1
func @not_cancellable_operations_1(%arg0: !async.token, %arg1: i1) {
  // Same reason as above, although `async.execute` is inside the nested
  // region or "regular" operation.
  //
  // NOTE: This test is not correct w.r.t. reference counting, and at runtime
  // would leak %arg0 value if %arg1 is false. IR like this will not be
  // constructed by automatic reference counting pass, because it would
  // place `async.add_ref` right before the `async.execute` inside `scf.if`.
  
  // CHECK: async.add_ref
  async.add_ref %arg0 {count = 1 : i32} : !async.token
  scf.if %arg1 {
    %token = async.execute {
      async.await %arg0 : !async.token
      // CHECK: async.drop_ref
      async.drop_ref %arg0 {count = 1 : i32} : !async.token
      async.yield
    }
  }
  // CHECK: async.await
  async.await %arg0 : !async.token
  // CHECK: async.drop_ref
  async.drop_ref %arg0 {count = 1 : i32} : !async.token
  // CHECK: return
  return
}

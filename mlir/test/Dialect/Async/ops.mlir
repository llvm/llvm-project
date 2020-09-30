// RUN: mlir-opt  %s | FileCheck %s

// CHECK-LABEL: @identity
func @identity(%arg0 : !async.token) -> !async.token {
  // CHECK: return %arg0 : !async.token
  return %arg0 : !async.token
}

// CHECK-LABEL: @empty_async_execute
func @empty_async_execute() -> !async.token {
  %0 = async.execute {
    async.yield
  } : !async.token

  return %0 : !async.token
}

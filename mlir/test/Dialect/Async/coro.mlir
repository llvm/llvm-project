// RUN: mlir-opt %s | FileCheck %s --dump-input=always

// CHECK-LABEL: @coro_id
func @coro_id() -> !async.coro.id {
  // CHECK: %0 = async.coro.id
  // CHECK: return %0 : !async.coro.id
  %0 = async.coro.id
  return %0 : !async.coro.id
}

// CHECK-LABEL: @coro_handle
func @coro_handle(%arg0: !async.coro.id) -> !async.coro.handle {
  // CHECK: %0 = async.coro.begin %arg0
  // CHECK: return %0 : !async.coro.handle
  %0 = async.coro.begin %arg0
  return %0 : !async.coro.handle
}

// CHECK-LABEL: @coro_free
func @coro_free(%arg0: !async.coro.id, %arg1: !async.coro.handle) {
  // CHECK: async.coro.free %arg0, %arg1
  async.coro.free %arg0, %arg1
  return
}

// CHECK-LABEL: @coro_end
func @coro_end(%arg0: !async.coro.handle) {
  // CHECK: async.coro.end %arg0
  async.coro.end %arg0
  return
}

// CHECK-LABEL: @coro_save
func @coro_save(%arg0: !async.coro.handle) -> !async.coro.state {
  // CHECK: %0 = async.coro.save %arg0
  %0 = async.coro.save %arg0
  // CHECK: return %0 : !async.coro.state
  return %0 : !async.coro.state
}

// CHECK-LABEL: @coro_suspend
func @coro_suspend(%arg0: !async.coro.state) {
  // CHECK:      async.coro.suspend %arg0
  // CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[CLEANUP:.*]]
  async.coro.suspend %arg0, ^suspend, ^resume, ^cleanup
^resume:
  // CHECK: ^[[RESUME]]
  // CHECK: return {coro.resume}
  return { coro.resume }
^cleanup:
  // CHECK: ^[[CLEANUP]]
  // CHECK: return {coro.cleanup}
  return { coro.cleanup }
^suspend:
  // CHECK: ^[[SUSPEND]]
  // CHECK: return {coro.suspend}
  return { coro.suspend }
}

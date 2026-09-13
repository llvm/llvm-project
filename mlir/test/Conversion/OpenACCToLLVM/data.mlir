// RUN: mlir-opt %s -acc-to-llvm -split-input-file | FileCheck %s

// A structured data region with a prepared map entry becomes a begin/end pair
// of runtime mapping calls. The map flags and size are taken from acc.map_info.
// CHECK-LABEL: llvm.func @data_copy
// CHECK-NOT: acc.data
// CHECK-NOT: acc.map_info
// CHECK-DAG: %[[TOFROM:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG: %[[SIZE:.*]] = llvm.mlir.constant(40 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.call @__tgt_acc_data_end
func.func @data_copy(%arg0: !llvm.ptr) {
  %size = arith.constant 40 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(f32)
      size(%size : i64) elementSize(4) name("a")
      descKind(none) mapFlags(to, from) -> !llvm.ptr
  acc.data dataOperands(%map : !llvm.ptr) {
    acc.terminator
  }
  return
}

// -----

// Enter and exit data use the unstructured runtime entry points.
// CHECK-LABEL: llvm.func @enter_exit
// CHECK: llvm.call @__tgt_acc_data_enter
// CHECK: llvm.call @__tgt_acc_data_exit
func.func @enter_exit(%arg0: !llvm.ptr) {
  %size = arith.constant 4 : i64
  %in = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) name("n")
      descKind(none) mapFlags(to) -> !llvm.ptr
  acc.enter_data dataOperands(%in : !llvm.ptr)
  %out = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) name("n")
      descKind(none) mapFlags(from) -> !llvm.ptr
  acc.exit_data dataOperands(%out : !llvm.ptr)
  return
}

// -----

// Update device uses the update runtime entry point.
// CHECK-LABEL: llvm.func @update_device
// CHECK: llvm.call @__tgt_acc_data_update
func.func @update_device(%arg0: !llvm.ptr) {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4)
      descKind(none) mapFlags(to) -> !llvm.ptr
  acc.update dataOperands(%map : !llvm.ptr)
  return
}

// -----

// Data-clause operations can be lowered directly when map_info preparation is
// not needed. Their clause semantics are packed into the same runtime arrays.
// CHECK-LABEL: llvm.func @data_clause_ops
// CHECK-NOT: acc.copyin
// CHECK-NOT: acc.copyout
// CHECK-DAG: %[[TOFROM:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG: %[[SIZE:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.call @__tgt_acc_data_end
func.func @data_clause_ops(%arg0: !llvm.ptr) {
  %copy = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f64)
      dataClause(acc_copy) name("a") -> !llvm.ptr
  acc.data dataOperands(%copy : !llvm.ptr) {
    acc.terminator
  }
  acc.copyout accPtr(%copy : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr)
      varType(f64) dataClause(acc_copy) name("a")
  return
}

// -----

// CHECK-LABEL: llvm.func @update_clause_op
// CHECK-NOT: acc.update_device
// CHECK: llvm.call @__tgt_acc_data_update
func.func @update_clause_op(%arg0: !llvm.ptr) {
  %device = acc.update_device varPtr(%arg0 : !llvm.ptr) varType(i32)
      name("n") -> !llvm.ptr
  acc.update dataOperands(%device : !llvm.ptr)
  return
}

// -----

// An async clause names the queue the mapping calls run on, and a wait clause
// makes them wait for the queues it names first. The wait is not part of the
// data region, so it is emitted once, before the region is entered.
// CHECK-LABEL: llvm.func @data_async_wait
// CHECK: %[[QUEUE:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: llvm.call @__tgt_acc_wait({{.*}}, %[[QUEUE]])
// CHECK: llvm.call @__tgt_acc_data_begin({{.*}}, %[[QUEUE]])
// CHECK: llvm.call @__tgt_acc_data_end({{.*}}, %[[QUEUE]])
func.func @data_async_wait(%arg0: !llvm.ptr) {
  %queue = arith.constant 2 : i64
  %size = arith.constant 40 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(f32)
      size(%size : i64) elementSize(4) name("a")
      descKind(none) mapFlags(to, from) -> !llvm.ptr
  acc.data dataOperands(%map : !llvm.ptr) async(%queue : i64)
      wait({%queue : i64}) {
    acc.terminator
  }
  return
}

// -----

// An async clause can be given once per device type. This pass lowers the
// clauses that apply when no device type is selected, so the queue is the one
// the default async clause names and the nvidia one is left unused.
// CHECK-LABEL: llvm.func @data_async_device_type
// CHECK: %[[QUEUE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin({{.*}}, %[[QUEUE]])
func.func @data_async_device_type(%arg0: !llvm.ptr) {
  %none = arith.constant 1 : i64
  %nvidia = arith.constant 7 : i64
  %size = arith.constant 40 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(f32)
      size(%size : i64) elementSize(4) name("a")
      descKind(none) mapFlags(to) -> !llvm.ptr
  acc.data dataOperands(%map : !llvm.ptr)
      async(%none : i64, %nvidia : i64 [#acc.device_type<nvidia>]) {
    acc.terminator
  }
  return
}

// -----

// An if clause guards both mapping calls of a data region: the region body
// runs either way, the mapping only happens when the condition holds.
// CHECK-LABEL: llvm.func @data_if
// CHECK: llvm.cond_br %arg1
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.cond_br %arg1
// CHECK: llvm.call @__tgt_acc_data_end
func.func @data_if(%arg0: !llvm.ptr, %cond: i1) {
  %size = arith.constant 40 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(f32)
      size(%size : i64) elementSize(4) name("a")
      descKind(none) mapFlags(to, from) -> !llvm.ptr
  acc.data if(%cond) dataOperands(%map : !llvm.ptr) {
    acc.terminator
  }
  return
}

// -----

// A wait clause without queues waits for all of them, which the runtime call
// states with an empty queue list.
// CHECK-LABEL: llvm.func @enter_data_wait_all
// CHECK: %[[WAITNUM:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WAITLIST:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @__tgt_acc_wait({{.*}}, %[[WAITNUM]], %[[WAITLIST]], {{.*}})
// CHECK: llvm.call @__tgt_acc_data_enter
func.func @enter_data_wait_all(%arg0: !llvm.ptr) {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) name("n")
      descKind(none) mapFlags(to) -> !llvm.ptr
  acc.enter_data dataOperands(%map : !llvm.ptr) wait
  return
}

// -----

// Other entry operations use the same runtime-argument emission and keep the
// flags of their clause: present, no_create, deviceptr and attach. The address
// of a device pointer is one the runtime must not look up, so it is handed over
// in a slot the runtime reads it from.
// CHECK-LABEL: llvm.func @entry_clause_ops
// CHECK-NOT: acc.present
// CHECK-NOT: acc.nocreate
// CHECK-NOT: acc.deviceptr
// CHECK-NOT: acc.attach
// CHECK-DAG: llvm.mlir.constant(1048576 : i64) : i64
// CHECK-DAG: llvm.mlir.constant(8192 : i64) : i64
// CHECK-DAG: llvm.mlir.constant(1024 : i64) : i64
// CHECK-DAG: llvm.mlir.constant(16 : i64) : i64
// CHECK-DAG: llvm.store %arg2, %{{.*}} : !llvm.ptr, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.call @__tgt_acc_data_end
func.func @entry_clause_ops(%arg0: !llvm.ptr, %arg1: !llvm.ptr,
    %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
  %present = acc.present varPtr(%arg0 : !llvm.ptr) varType(f64) -> !llvm.ptr
  %noCreate = acc.nocreate varPtr(%arg1 : !llvm.ptr) varType(f64)
      -> !llvm.ptr
  %devicePtr = acc.deviceptr varPtr(%arg2 : !llvm.ptr) varType(f64)
      -> !llvm.ptr
  %attach = acc.attach varPtr(%arg3 : !llvm.ptr) varType(f64) -> !llvm.ptr
  acc.data dataOperands(%present, %noCreate, %devicePtr, %attach
      : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
    acc.terminator
  }
  return
}

// -----

// Each unstructured data construct applies its own if condition to the runtime
// call, without guarding the surrounding function.
// CHECK-LABEL: llvm.func @unstructured_data_if
// CHECK: llvm.cond_br %arg1
// CHECK: llvm.call @__tgt_acc_data_enter
// CHECK: llvm.cond_br %arg1
// CHECK: llvm.call @__tgt_acc_data_update
// CHECK: llvm.cond_br %arg1
// CHECK: llvm.call @__tgt_acc_data_exit
// CHECK: llvm.return
func.func @unstructured_data_if(%arg0: !llvm.ptr, %cond: i1) {
  %size = arith.constant 4 : i64
  %enter = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(to) -> !llvm.ptr
  acc.enter_data dataOperands(%enter : !llvm.ptr) if(%cond)
  %update = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(from) -> !llvm.ptr
  acc.update dataOperands(%update : !llvm.ptr) if(%cond)
  %exit = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(from) -> !llvm.ptr
  acc.exit_data dataOperands(%exit : !llvm.ptr) if(%cond)
  return
}

// -----

// Data constructs can form one async chain. Every wait precedes the operation
// it constrains, and each operation receives its own queue.
// CHECK-LABEL: llvm.func @data_async_chain
// CHECK: %[[Q100:.*]] = llvm.mlir.constant(100 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin({{.*}}, %[[Q100]])
// CHECK: %[[Q200:.*]] = llvm.mlir.constant(200 : i64) : i64
// CHECK: %[[W100:.*]] = llvm.mlir.constant(100 : i64) : i64
// CHECK: llvm.store %[[W100]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_wait({{.*}}, %[[Q200]])
// CHECK: llvm.call @__tgt_acc_data_update({{.*}}, %[[Q200]])
// CHECK: llvm.call @__tgt_acc_data_end({{.*}}, %[[Q100]])
// CHECK: %[[Q300:.*]] = llvm.mlir.constant(300 : i64) : i64
// CHECK: %[[W100_ENTER:.*]] = llvm.mlir.constant(100 : i64) : i64
// CHECK: llvm.store %[[W100_ENTER]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_wait({{.*}}, %[[Q300]])
// CHECK: llvm.call @__tgt_acc_data_enter({{.*}}, %[[Q300]])
// CHECK: %[[Q400:.*]] = llvm.mlir.constant(400 : i64) : i64
// CHECK: %[[W300:.*]] = llvm.mlir.constant(300 : i64) : i64
// CHECK: llvm.store %[[W300]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_wait({{.*}}, %[[Q400]])
// CHECK: llvm.call @__tgt_acc_data_exit({{.*}}, %[[Q400]])
func.func @data_async_chain(%arg0: !llvm.ptr) {
  %q100 = arith.constant 100 : i32
  %q200 = arith.constant 200 : i32
  %q300 = arith.constant 300 : i32
  %q400 = arith.constant 400 : i32
  %size = arith.constant 4 : i64
  %regionMap = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(to, from) -> !llvm.ptr
  acc.data dataOperands(%regionMap : !llvm.ptr) async(%q100 : i32) {
    %updateMap = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
        size(%size : i64) elementSize(4) descKind(none)
        mapFlags(from) -> !llvm.ptr
    acc.update dataOperands(%updateMap : !llvm.ptr) async(%q200 : i32)
        wait({%q100 : i32})
    acc.terminator
  }
  %enterMap = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(to) -> !llvm.ptr
  acc.enter_data dataOperands(%enterMap : !llvm.ptr) async(%q300 : i32)
      wait(%q100 : i32)
  %exitMap = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(from) -> !llvm.ptr
  acc.exit_data dataOperands(%exitMap : !llvm.ptr) async(%q400 : i32)
      wait(%q300 : i32)
  return
}

// -----

// Async queue operands are normalized to the runtime's i64 queue type.
// CHECK-LABEL: llvm.func @data_async_integer_widths
// CHECK: %[[Q32:.*]] = llvm.sext %arg1 : i32 to i64
// CHECK: llvm.call @__tgt_acc_data_enter({{.*}}, %[[Q32]])
// CHECK: %[[Q16:.*]] = llvm.sext %arg2 : i16 to i64
// CHECK: llvm.call @__tgt_acc_data_exit({{.*}}, %[[Q16]])
func.func @data_async_integer_widths(%arg0: !llvm.ptr, %q32: i32, %q16: i16) {
  %size = arith.constant 4 : i64
  %enter = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(to) -> !llvm.ptr
  acc.enter_data dataOperands(%enter : !llvm.ptr) async(%q32 : i32)
  %exit = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(from) -> !llvm.ptr
  acc.exit_data dataOperands(%exit : !llvm.ptr) async(%q16 : i16)
  return
}

// -----

// An async clause without a queue uses the runtime's no-value sentinel.
// CHECK-LABEL: llvm.func @data_async_noval
// CHECK: %[[NOVAL:.*]] = llvm.mlir.constant(-4 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin({{.*}}, %[[NOVAL]])
// CHECK: llvm.call @__tgt_acc_data_end({{.*}}, %[[NOVAL]])
func.func @data_async_noval(%arg0: !llvm.ptr) {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(to, from) -> !llvm.ptr
  acc.data dataOperands(%map : !llvm.ptr) async {
    acc.terminator
  }
  return
}

// -----

// Update waits for the named queues before the mapping call.
// CHECK-LABEL: llvm.func @update_wait
// CHECK: %[[QUEUE:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK: llvm.store %[[QUEUE]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_wait
// CHECK: llvm.call @__tgt_acc_data_update
func.func @update_wait(%arg0: !llvm.ptr) {
  %q = arith.constant 7 : i32
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(from) -> !llvm.ptr
  acc.update dataOperands(%map : !llvm.ptr) wait({%q : i32})
  return
}

// RUN: mlir-opt %s -acc-to-llvm -split-input-file | FileCheck %s

// A structured declare (the enter token is consumed by a matching exit) uses
// the same begin/end runtime entry points as acc.data, with the declare
// region flag (2) so the runtime can tell the two apart.
// CHECK-LABEL: llvm.func @structured_declare
// CHECK: %[[BEGIN_FLAGS:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin(%{{[^,]+}}, %[[BEGIN_FLAGS]],
// CHECK: %[[END_FLAGS:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_end(%{{[^,]+}}, %[[END_FLAGS]],
// CHECK-NOT: acc.declare_enter
// CHECK-NOT: acc.declare_exit
func.func @structured_declare(%arg0: !llvm.ptr) {
  %size = arith.constant 40 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(f32)
      size(%size : i64) elementSize(4) name("a")
      descKind(none) mapFlags(none) -> !llvm.ptr
  %token = acc.declare_enter dataOperands(%map : !llvm.ptr)
  acc.declare_exit token(%token) dataOperands(%map : !llvm.ptr)
  return
}

// -----

// A structured declare whose exit names no objects, as a declare of a device
// pointer does, still opens its region with the declare flag; there is no
// end call, as nothing is to be released.
// CHECK-LABEL: llvm.func @structured_declare_no_exit_operands
// CHECK: %[[BEGIN_FLAGS:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin(%{{[^,]+}}, %[[BEGIN_FLAGS]],
// CHECK-NOT: llvm.call @__tgt_acc_data_end
// CHECK-NOT: acc.declare_exit
func.func @structured_declare_no_exit_operands(%arg0: !llvm.ptr) {
  %size = arith.constant 40 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(f32)
      size(%size : i64) elementSize(4) name("a")
      descKind(none) mapFlags(devptr) -> !llvm.ptr
  %token = acc.declare_enter dataOperands(%map : !llvm.ptr)
  acc.declare_exit token(%token)
  return
}

// -----

// An unstructured declare registers the objects with __tgt_acc_declare. With
// no target hook installed, its binary descriptor is null.
// CHECK-LABEL: llvm.func @unstructured_declare
// CHECK: %[[DESC:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @__tgt_acc_declare({{.*}}%[[DESC]])
// CHECK-NOT: acc.declare_enter
func.func @unstructured_declare(%arg0: !llvm.ptr) {
  %size = arith.constant 28 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(f32)
      size(%size : i64) elementSize(4) name("arr")
      descKind(none) mapFlags(none) -> !llvm.ptr
  %token = acc.declare_enter dataOperands(%map : !llvm.ptr)
  return
}

// -----

// A recipe that runs after the host object is allocated only allocates the
// device mirror of it.
// CHECK-LABEL: llvm.func @alloc_recipe
// CHECK: llvm.call @__tgt_acc_mirror_alloc
// CHECK-NOT: llvm.call @__tgt_acc_data_begin
// CHECK-NOT: llvm.call @__tgt_acc_declare
llvm.func @alloc_recipe(%arg0: !llvm.ptr) attributes {acc.declare_action = #acc.declare_action<postAlloc = @alloc_recipe>} {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4)
      descKind(none) mapFlags(none) -> !llvm.ptr
  %token = acc.declare_enter dataOperands(%map : !llvm.ptr)
  llvm.return
}

// -----

// A recipe that runs before the host object is deallocated frees the device
// mirror of it.
// CHECK-LABEL: llvm.func @pre_dealloc_recipe
// CHECK: llvm.call @__tgt_acc_mirror_dealloc
// CHECK-NOT: llvm.call @__tgt_acc_data_end
llvm.func @pre_dealloc_recipe(%arg0: !llvm.ptr) attributes {acc.declare_action = #acc.declare_action<preDealloc = @pre_dealloc_recipe>} {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4)
      descKind(none) mapFlags(delete) -> !llvm.ptr
  acc.declare_exit dataOperands(%map : !llvm.ptr)
  llvm.return
}

// -----

// A recipe that runs after the host object is deallocated releases the
// mapping of the descriptor, which is still valid storage at that point. The
// exit holds no token, so the call carries no declare region flag.
// CHECK-LABEL: llvm.func @post_dealloc_recipe
// CHECK: llvm.mlir.addressof @acc.ident
// CHECK: %[[END_FLAGS:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_end(%{{[^,]+}}, %[[END_FLAGS]],
// CHECK-NOT: llvm.call @__tgt_acc_mirror_dealloc
llvm.func @post_dealloc_recipe(%arg0: !llvm.ptr) attributes {acc.declare_action = #acc.declare_action<postDealloc = @post_dealloc_recipe>} {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4)
      descKind(none) mapFlags(delete) -> !llvm.ptr
  acc.declare_exit dataOperands(%map : !llvm.ptr)
  llvm.return
}

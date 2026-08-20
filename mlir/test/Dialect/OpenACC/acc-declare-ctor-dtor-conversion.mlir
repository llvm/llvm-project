// RUN: mlir-opt %s -acc-declare-ctor-dtor-conversion -split-input-file | FileCheck %s
// RUN: mlir-opt %s --pass-pipeline="builtin.module(acc-declare-ctor-dtor-conversion{generate-dtors=false})" -split-input-file | FileCheck %s --check-prefix=NODTOR

// CHECK-NOT: acc.global_ctor
// CHECK-NOT: acc.global_dtor
// CHECK: llvm.func internal @arr_acc_ctor()
// CHECK: llvm.func internal @other_arr_acc_ctor()
// CHECK: llvm.func internal @arr_acc_dtor()
// CHECK: llvm.func internal @other_arr_acc_dtor()
// CHECK: llvm.mlir.global_ctors ctors = [@arr_acc_ctor, @other_arr_acc_ctor], priorities = [102 : i32, 102 : i32], data = [#llvm.zero, #llvm.zero]
// CHECK: llvm.mlir.global_dtors dtors = [@arr_acc_dtor, @other_arr_acc_dtor], priorities = [102 : i32, 102 : i32], data = [#llvm.zero, #llvm.zero]

// NODTOR-NOT: @arr_acc_dtor
// NODTOR-NOT: @other_arr_acc_dtor
// NODTOR: llvm.func internal @arr_acc_ctor()
// NODTOR: llvm.func internal @other_arr_acc_ctor()
// NODTOR: llvm.mlir.global_ctors ctors = [@arr_acc_ctor, @other_arr_acc_ctor], priorities = [102 : i32, 102 : i32], data = [#llvm.zero, #llvm.zero]
// NODTOR-NOT: llvm.mlir.global_dtors

llvm.mlir.global external @arr() {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.array<7 x f32> {
  %0 = llvm.mlir.zero : !llvm.array<7 x f32>
  llvm.return %0 : !llvm.array<7 x f32>
}
llvm.mlir.global external @other_arr() {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.array<3 x f32> {
  %0 = llvm.mlir.zero : !llvm.array<3 x f32>
  llvm.return %0 : !llvm.array<3 x f32>
}
acc.global_ctor @arr_acc_ctor {
  %0 = llvm.mlir.addressof @arr {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.ptr
  %1 = acc.create varPtr(%0 : !llvm.ptr) varType(!llvm.array<7 x f32>) -> !llvm.ptr
  acc.declare_enter dataOperands(%1 : !llvm.ptr)
  acc.terminator
}
acc.global_ctor @other_arr_acc_ctor {
  %0 = llvm.mlir.addressof @other_arr {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.ptr
  %1 = acc.create varPtr(%0 : !llvm.ptr) varType(!llvm.array<3 x f32>) -> !llvm.ptr
  acc.declare_enter dataOperands(%1 : !llvm.ptr)
  acc.terminator
}
acc.global_dtor @arr_acc_dtor {
  %0 = llvm.mlir.addressof @arr {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.ptr
  %1 = acc.getdeviceptr varPtr(%0 : !llvm.ptr) varType(!llvm.array<7 x f32>) -> !llvm.ptr {dataClause = #acc<data_clause acc_create>}
  acc.declare_exit dataOperands(%1 : !llvm.ptr)
  acc.delete accPtr(%1 : !llvm.ptr)
  acc.terminator
}
acc.global_dtor @other_arr_acc_dtor {
  %0 = llvm.mlir.addressof @other_arr {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.ptr
  %1 = acc.getdeviceptr varPtr(%0 : !llvm.ptr) varType(!llvm.array<3 x f32>) -> !llvm.ptr {dataClause = #acc<data_clause acc_create>}
  acc.declare_exit dataOperands(%1 : !llvm.ptr)
  acc.delete accPtr(%1 : !llvm.ptr)
  acc.terminator
}

// -----

// Merge with an existing llvm.mlir.global_ctors entry.

// CHECK: llvm.func internal @existing_ctor
// CHECK: llvm.func internal @merged_acc_ctor
// CHECK: llvm.mlir.global_ctors ctors = [@existing_ctor, @merged_acc_ctor], priorities = [0 : i32, 102 : i32], data = [#llvm.zero, #llvm.zero]

llvm.mlir.global external @merged_var() {acc.declare = #acc.declare<dataClause = acc_create>} : i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}
llvm.func internal @existing_ctor() {
  llvm.return
}
llvm.mlir.global_ctors ctors = [@existing_ctor], priorities = [0 : i32], data = [#llvm.zero]
acc.global_ctor @merged_acc_ctor {
  %0 = llvm.mlir.addressof @merged_var {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.ptr
  %1 = acc.create varPtr(%0 : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare_enter dataOperands(%1 : !llvm.ptr)
  acc.terminator
}

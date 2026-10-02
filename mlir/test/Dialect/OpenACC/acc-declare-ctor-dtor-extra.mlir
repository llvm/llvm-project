// RUN: mlir-opt %s --pass-pipeline='builtin.module(acc-declare-ctor-dtor-conversion{extra-constructors=foo entry-only-constructors=bar entry-point-name=main})' -split-input-file | FileCheck %s

// The functions in extra-constructors are always declared and called from
// acc.extr_ctor (or given name in extra-ctor-name option);
// The functions in entry-only-constructors are declared and called from
// the extra ctor if the entry-point-name is present in the module.

// CHECK: llvm.func @foo()
// CHECK: llvm.func internal @acc.extra_ctor() {
// CHECK:   llvm.call @foo() : () -> ()
// CHECK-NOT: llvm.call @bar()
// CHECK:   llvm.return
// CHECK: }
// CHECK: llvm.mlir.global_ctors ctors = [@acc.extra_ctor], priorities = [102 : i32], data = [#llvm.zero]
// CHECK-NOT: llvm.func @bar()

module {
}

// -----

// CHECK: llvm.func @foo()
// CHECK: llvm.func @bar()
// CHECK: llvm.func internal @acc.extra_ctor() {
// CHECK:   llvm.call @foo() : () -> ()
// CHECK:   llvm.call @bar() : () -> ()
// CHECK:   llvm.return
// CHECK: }
// CHECK: llvm.mlir.global_ctors ctors = [@acc.extra_ctor], priorities = [102 : i32], data = [#llvm.zero]

module {
  llvm.func @main() {
    llvm.return
  }
}

// -----

// An existing declaration is reused. The extra constructor is merged into
// llvm.mlir.global_ctors next to converted acc.global_ctor functions.

// CHECK: llvm.func @foo()
// CHECK-NOT: llvm.func @foo()
// CHECK: llvm.func internal @arr_acc_ctor()
// CHECK: llvm.func internal @acc.extra_ctor() {
// CHECK:   llvm.call @foo() : () -> ()
// CHECK:   llvm.return
// CHECK: }
// CHECK: llvm.mlir.global_ctors ctors = [@arr_acc_ctor, @acc.extra_ctor], priorities = [102 : i32, 102 : i32], data = [#llvm.zero, #llvm.zero]

llvm.mlir.global external @arr() {acc.declare = #acc.declare<dataClause = acc_create>} : i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}
llvm.func @foo()
acc.global_ctor @arr_acc_ctor {
  %0 = llvm.mlir.addressof @arr {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.ptr
  %1 = acc.create varPtr(%0 : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare_enter dataOperands(%1 : !llvm.ptr)
  acc.terminator
}

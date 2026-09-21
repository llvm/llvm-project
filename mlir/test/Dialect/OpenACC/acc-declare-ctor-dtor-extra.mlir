// RUN: mlir-opt %s --pass-pipeline='builtin.module(acc-declare-ctor-dtor-conversion{extra-constructors=foo entry-only-constructors=bar entry-point-name=main})' -split-input-file | FileCheck %s

// extra-constructors is a list of <name>:<if-main> pairs. The functions are
// declared and called from __openaccExtraConstructor; llvm.mlir.global_ctors
// cannot reference a declaration. if-main=false always calls the function;
// if-main=true calls it only when the module contains program-entry-name.

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

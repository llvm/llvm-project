// RUN: mlir-opt --pass-pipeline='builtin.module(llvm.func(openacc-legalize-data-values))' %s | FileCheck %s

// The pass is anchored on FunctionOpInterface, so it also runs on function
// operations other than func.func (e.g. llvm.func).

// CHECK-LABEL: llvm.func @test
// CHECK-SAME: (%[[A:.*]]: !llvm.ptr)
// CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : !llvm.ptr) varType(f32) -> !llvm.ptr
// CHECK: acc.parallel dataOperands(%[[CREATE]] : !llvm.ptr) {
// CHECK:   %{{.*}} = llvm.load %[[CREATE]] : !llvm.ptr -> f32
llvm.func @test(%a: !llvm.ptr) {
  %create = acc.create varPtr(%a : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.parallel dataOperands(%create : !llvm.ptr) {
    %v = llvm.load %a : !llvm.ptr -> f32
    acc.yield
  }
  llvm.return
}

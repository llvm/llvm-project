// RUN: mlir-opt %s -inline -mlir-print-debuginfo | FileCheck %s

llvm.func @foo() -> !llvm.ptr

llvm.func @with_byval_arg(%ptr : !llvm.ptr { llvm.byval = f64 }) {
  llvm.return
}

// CHECK-LABEL: llvm.func @test_byval
llvm.func @test_byval() {
  // CHECK: %[[COPY:.+]] = llvm.alloca %{{.+}} x f64
  // CHECK-SAME: loc(#[[LOC:.+]])
  // CHECK: %[[ORIG:.+]] = llvm.call @foo() : () -> !llvm.ptr loc(#[[LOC]])
  %0 = llvm.call @foo() : () -> !llvm.ptr loc("inlining-debuginfo.mlir":14:2)
  // CHECK: "llvm.intr.memcpy"(%[[COPY]], %[[ORIG]]
  // CHECK-SAME: loc(#[[LOC]])
  llvm.call @with_byval_arg(%0) : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK: #[[LOC]] = loc("inlining-debuginfo.mlir":14:2)

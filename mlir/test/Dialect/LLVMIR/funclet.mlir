// RUN: mlir-opt %s | mlir-opt | FileCheck %s

llvm.func @foo()

// CHECK-LABEL: @test_cleanup
llvm.func @test_cleanup() {
  %none = llvm.mlir.none : token
  // CHECK: %[[CP:.*]] = llvm.cleanuppad within %{{.*}}
  %cp = llvm.cleanuppad within %none
  // CHECK: llvm.call @foo() ["funclet"(%[[CP]] : token)] : () -> ()
  llvm.call @foo() ["funclet"(%cp : token)] : () -> ()
  // CHECK: llvm.cleanupret %[[CP]]
  llvm.cleanupret %cp
}

// CHECK-LABEL: @test_cleanup_args
llvm.func @test_cleanup_args(%arg0: !llvm.ptr) {
  %none = llvm.mlir.none : token
  // CHECK: %[[CP:.*]] = llvm.cleanuppad within %{{.*}}[%{{.*}} : !llvm.ptr]
  %cp = llvm.cleanuppad within %none [%arg0 : !llvm.ptr]
  // CHECK: llvm.cleanupret %[[CP]]
  llvm.cleanupret %cp
}

// CHECK-LABEL: @test_cleanup_unwind
llvm.func @test_cleanup_unwind() {
  %none = llvm.mlir.none : token
  // CHECK: %[[CP:.*]] = llvm.cleanuppad within %{{.*}}
  %cp = llvm.cleanuppad within %none
  // CHECK: llvm.cleanupret %[[CP]] unwind ^[[BB:.*]]
  llvm.cleanupret %cp unwind ^unwind_bb
^unwind_bb:
  llvm.return
}

// CHECK-LABEL: @test_catch
llvm.func @test_catch(%arg0: !llvm.ptr) {
  %none = llvm.mlir.none : token
  // CHECK: %[[CS:.*]] = llvm.catchswitch within %{{.*}} [^[[H:.*]]] unwind to caller
  %cs = llvm.catchswitch within %none [^catch_body] unwind to caller
^catch_body:
  // CHECK: %[[PAD:.*]] = llvm.catchpad within %[[CS]][%{{.*}} : !llvm.ptr]
  %pad = llvm.catchpad within %cs [%arg0 : !llvm.ptr]
  // CHECK: llvm.call @foo() ["funclet"(%[[PAD]] : token)] : () -> ()
  llvm.call @foo() ["funclet"(%pad : token)] : () -> ()
  // CHECK: llvm.catchret from %[[PAD]] to ^[[EXIT:.*]]
  llvm.catchret from %pad to ^exit
^exit:
  llvm.return
}

// CHECK-LABEL: @test_catch_unwind
llvm.func @test_catch_unwind(%arg0: !llvm.ptr) {
  %none = llvm.mlir.none : token
  // CHECK: %[[CS:.*]] = llvm.catchswitch within %{{.*}} [^[[H:.*]]] unwind ^[[UNWIND:.*]]
  %cs = llvm.catchswitch within %none [^catch_body] unwind ^unwind_bb
^catch_body:
  // CHECK: %[[PAD:.*]] = llvm.catchpad within %[[CS]][%{{.*}} : !llvm.ptr]
  %pad = llvm.catchpad within %cs [%arg0 : !llvm.ptr]
  // CHECK: llvm.catchret from %[[PAD]] to ^[[EXIT:.*]]
  llvm.catchret from %pad to ^exit
^unwind_bb:
  llvm.return
^exit:
  llvm.return
}

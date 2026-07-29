// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false, omp.is_gpu = false, omp.version = #omp.version<version = 31>} {
  // CHECK-LABEL: define void @_QMfuncsPfoo_variant()
  llvm.func @_QMfuncsPfoo_variant() {
    llvm.return
  }
  // CHECK-LABEL: define void @_QMfuncsPfoo_dispatch()
  llvm.func @_QMfuncsPfoo_dispatch() {
    llvm.return
  }
  // CHECK-LABEL: define void @_QQmain()
  llvm.func @_QQmain() {
    // CHECK: call void @_QMfuncsPfoo_dispatch()
    llvm.call @_QMfuncsPfoo_dispatch() : () -> ()
    // CHECK: br label %omp.dispatch.region
    // CHECK: omp.dispatch.region:
    omp.dispatch {
      // CHECK: call void @_QMfuncsPfoo_variant()
      llvm.call @_QMfuncsPfoo_variant() : () -> ()
      // CHECK: br label %omp.region.cont
      omp.terminator
    }
    // CHECK: omp.region.cont:
    llvm.return
  }
  // The nowait clause is accepted; it is a no-op in the current synchronous
  // inline lowering, producing the same dispatch region.
  // CHECK-LABEL: define void @test_dispatch_nowait()
  llvm.func @test_dispatch_nowait() {
    // CHECK: br label %omp.dispatch.region
    // CHECK: omp.dispatch.region:
    omp.dispatch nowait {
      // CHECK: call void @_QMfuncsPfoo_variant()
      llvm.call @_QMfuncsPfoo_variant() : () -> ()
      // CHECK: br label %omp.region.cont
      omp.terminator
    }
    // CHECK: omp.region.cont:
    llvm.return
  }
  // The novariants operand is ignored at translation; the region already holds
  // the runtime base/variant selection.
  // CHECK-LABEL: define void @test_dispatch_novariants(i1
  llvm.func @test_dispatch_novariants(%cond : i1) {
    // CHECK: br label %omp.dispatch.region
    // CHECK: omp.dispatch.region:
    omp.dispatch novariants(%cond) {
      // CHECK: call void @_QMfuncsPfoo_variant()
      llvm.call @_QMfuncsPfoo_variant() : () -> ()
      // CHECK: br label %omp.region.cont
      omp.terminator
    }
    // CHECK: omp.region.cont:
    llvm.return
  }
}

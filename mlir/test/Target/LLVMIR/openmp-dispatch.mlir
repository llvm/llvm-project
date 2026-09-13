// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

module attributes {omp.is_target_device = false, omp.is_gpu = false, omp.version = #omp.version<version = 51>} {
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
    // The producer of the region has already materialized the base/variant
    // selection; MLIR only translates the call inside the region.
    omp.dispatch {
      // CHECK: call void @_QMfuncsPfoo_variant()
      llvm.call @_QMfuncsPfoo_variant() : () -> ()
      // CHECK: br label %omp.region.cont
      omp.terminator
    }
    // CHECK: omp.region.cont:
    llvm.return
  }
  // The producer of the region materializes the base/variant selection; the
  // LLVM IR translation deliberately ignores the novariants operand.
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
  // The producer of the region materializes the base/variant selection; the
  // LLVM IR translation deliberately ignores the nocontext operand.
  // CHECK-LABEL: define void @test_dispatch_nocontext(i1
  llvm.func @test_dispatch_nocontext(%cond : i1) {
    // CHECK: br label %omp.dispatch.region
    // CHECK: omp.dispatch.region:
    omp.dispatch nocontext(%cond) {
      // CHECK: call void @_QMfuncsPfoo_variant()
      llvm.call @_QMfuncsPfoo_variant() : () -> ()
      // CHECK: br label %omp.region.cont
      omp.terminator
    }
    // CHECK: omp.region.cont:
    llvm.return
  }
}

// -----

// In OpenMP 5.2 and later `nowait` has no effect on `dispatch`; it is accepted
// and translated as a no-op (contrast openmp-todo.mlir, which diagnoses it for
// OpenMP < 5.2). Version 52 is used here as a representative of all >= 5.2.
module attributes {omp.is_target_device = false, omp.is_gpu = false, omp.version = #omp.version<version = 52>} {
  llvm.func @variant()
  // CHECK-LABEL: define void @dispatch_nowait_52()
  llvm.func @dispatch_nowait_52() {
    // CHECK: br label %omp.dispatch.region
    // CHECK: omp.dispatch.region:
    omp.dispatch nowait {
      // CHECK: call void @variant()
      llvm.call @variant() : () -> ()
      // CHECK: br label %omp.region.cont
      omp.terminator
    }
    // CHECK: omp.region.cont:
    llvm.return
  }
}

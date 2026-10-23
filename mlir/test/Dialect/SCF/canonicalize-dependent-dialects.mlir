// RUN: mlir-opt %s --canonicalize | FileCheck %s

// Canonicalizing a multi-block `scf.execute_region` materializes `cf.br` ops,
// so the SCF dialect has to declare a dependency on the ControlFlow dialect.
// This test deliberately contains no `cf` operation and does not use
// `func.func`: parsing a `cf` operation, or loading the Func dialect (whose
// inliner extension loads ControlFlow), would load the dialect for unrelated
// reasons and mask the missing dependency.

// CHECK-LABEL: llvm.func @multi_block_execute_region
//   CHECK-NOT:   scf.execute_region
//       CHECK:   llvm.cond_br
//       CHECK:   llvm.store
//       CHECK:   cf.br
//       CHECK:   cf.br
//       CHECK:   llvm.return
llvm.func @multi_block_execute_region(%c: i1, %p: !llvm.ptr, %x: i64) {
  scf.execute_region {
    llvm.cond_br %c, ^bb1, ^bb2
  ^bb1:
    llvm.store %x, %p : i64, !llvm.ptr
    scf.yield
  ^bb2:
    scf.yield
  }
  llvm.return
}

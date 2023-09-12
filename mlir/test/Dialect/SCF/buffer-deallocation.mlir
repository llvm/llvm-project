// DEFINE: %{canonicalize} = -canonicalize=enable-patterns="bufferization-skip-extract-metadata-of-alloc,bufferization-erase-always-false-dealloc,bufferization-erase-empty-dealloc,bufferization-dealloc-remove-duplicate-retained-memrefs,bufferization-dealloc-remove-duplicate-dealloc-memrefs",region-simplify=false

// RUN: mlir-opt -verify-diagnostics -buffer-deallocation \
// RUN:   %{canonicalize} -buffer-deallocation-simplification %{canonicalize} -split-input-file %s | FileCheck %s

func.func @parallel_insert_slice_no_conflict(%arg0: index) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<2xf32>
  scf.forall (%arg1) in (%arg0) {
    %alloc0 = memref.alloc() : memref<2xf32>
    %0 = memref.load %alloc[%c0] : memref<2xf32>
    linalg.fill ins(%0 : f32) outs(%alloc0 : memref<2xf32>)
  }
  return
}

// CHECK-LABEL: func @parallel_insert_slice_no_conflict
//  CHECK-SAME: (%arg0: index)
//       CHECK: [[ALLOC0:%.+]] = memref.alloc(
//       CHECK: scf.forall
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc(
//       CHECK:   bufferization.dealloc ([[ALLOC1]] : memref<2xf32>) if (%true
//   CHECK-NOT: retain
//       CHECK: }
//       CHECK: bufferization.dealloc ([[ALLOC0]] : memref<2xf32>) if (%true
//   CHECK-NOT: retain

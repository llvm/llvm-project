// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate="fast-mem-space=0" | FileCheck %s --check-prefixes=CHECK,PASS
// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate="generate-dma fast-mem-space=2" | FileCheck %s --check-prefixes=CHECK,PASS
// RUN: mlir-opt %s -split-input-file -test-affine-data-copy | FileCheck %s --check-prefixes=CHECK,PASS
// RUN: mlir-opt %s -split-input-file -test-affine-data-copy=memref-filter | FileCheck %s --check-prefixes=CHECK,FILTER

// A dynamic guard makes the access valid only when the index is zero. Do not
// speculate either copy-in or copy-out across the conditional inside the loop.
// CHECK-LABEL: func.func @guarded_loop(
// CHECK-SAME:    %[[MEM:.*]]: memref<1xi32>, %[[INDEX:.*]]: index)
// CHECK-NOT:     memref.alloc
// CHECK:         affine.for %{{.*}} = 0 to 4 {
// CHECK-NEXT:      affine.if #{{.*}}()[%[[INDEX]]] {
// CHECK-NEXT:        %[[VALUE:.*]] = affine.load %[[MEM]][symbol(%[[INDEX]])] : memref<1xi32>
// CHECK-NEXT:        affine.store %[[VALUE]], %[[MEM]][0] : memref<1xi32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func @guarded_loop(%m: memref<1xi32>, %index: index) {
  affine.for %i = 0 to 4 {
    affine.if affine_set<()[s0] : (s0 == 0)>()[%index] {
      %v = affine.load %m[symbol(%index)] : memref<1xi32>
      affine.store %v, %m[0] : memref<1xi32>
    }
  }
  return
}

// -----

// A conditional access to a memref excluded by the filter must not prevent
// copying an unconditionally accessed memref.
// CHECK-LABEL: func.func @filtered_memref(
// CHECK-SAME:    %[[MEM:.*]]: memref<8xi32>, %[[GUARDED:.*]]: memref<1xi32>, %[[INDEX:.*]]: index)
// CHECK-NOT:     memref.alloc
// CHECK:         affine.for %[[IV:.*]] = 0 to 8 {
// PASS-NEXT:       %[[VALUE:.*]] = affine.load %[[MEM]][%[[IV]]] : memref<8xi32>
// FILTER:          %[[FAST:.*]] = memref.alloc() : memref<1xi32>
// FILTER:          affine.load %[[MEM]][%[[IV]]] : memref<8xi32>
// FILTER:          affine.store %{{.*}}, %[[FAST]][0] : memref<1xi32>
// FILTER-NEXT:     %[[VALUE:.*]] = affine.load %[[FAST]][0] : memref<1xi32>
// CHECK-NEXT:      affine.if #{{.*}}()[%[[INDEX]]] {
// CHECK-NEXT:        %{{.*}} = affine.load %[[GUARDED]][symbol(%[[INDEX]])] : memref<1xi32>
// CHECK-NEXT:      }
// PASS-NEXT:       affine.store %[[VALUE]], %[[MEM]][%[[IV]]] : memref<8xi32>
// FILTER-NEXT:     affine.store %[[VALUE]], %[[FAST]][0] : memref<1xi32>
// FILTER:          affine.load %[[FAST]]
// FILTER:          affine.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<8xi32>
// FILTER:          memref.dealloc %[[FAST]] : memref<1xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func @filtered_memref(%m: memref<8xi32>, %guarded: memref<1xi32>, %index: index) {
  affine.for %i = 0 to 8 {
    %v = affine.load %m[%i] : memref<8xi32>
    affine.if affine_set<()[s0] : (s0 == 0)>()[%index] {
      %unused = affine.load %guarded[symbol(%index)] : memref<1xi32>
    }
    affine.store %v, %m[%i] : memref<8xi32>
  }
  return
}

// -----

// A known inner condition does not remove the need to check the outer guard.
// CHECK-LABEL: func.func @unknown_outer_true_inner(
// CHECK-SAME:    %[[MEM:.*]]: memref<1xi32>, %[[INDEX:.*]]: index, %[[VALUE:.*]]: i32)
// CHECK-NOT:     memref.alloc
// CHECK:         affine.for %{{.*}} = 0 to 4 {
// CHECK-NEXT:      affine.if #{{.*}}()[%[[INDEX]]] {
// CHECK-NEXT:        affine.if #{{.*}}() {
// CHECK-NEXT:          affine.store %[[VALUE]], %[[MEM]][0] : memref<1xi32>
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.store %[[VALUE]], %[[MEM]][8] : memref<1xi32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func @unknown_outer_true_inner(%m: memref<1xi32>, %index: index, %v: i32) {
  affine.for %i = 0 to 4 {
    affine.if affine_set<()[s0] : (s0 == 0)>()[%index] {
      affine.if affine_set<() : (1 >= 0)>() {
        affine.store %v, %m[0] : memref<1xi32>
      } else {
        affine.store %v, %m[8] : memref<1xi32>
      }
    }
  }
  return
}

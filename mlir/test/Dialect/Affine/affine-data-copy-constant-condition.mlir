// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate="fast-mem-space=0" | FileCheck %s --check-prefixes=CHECK,POINT
// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate="generate-dma fast-mem-space=2" | FileCheck %s --check-prefixes=CHECK,DMA

// The unreachable load must not enlarge the copy region of the live store.
// Only element zero is copied out; no copy-in is needed.
// CHECK-LABEL: func.func @unreachable_load()
// DMA:           %[[COUNT:.*]] = arith.constant 1 : index
// CHECK:         %[[MEM:.*]] = memref.alloc() : memref<1xi32>
// CHECK-NEXT:    %[[VALUE:.*]] = arith.constant 42 : i32
// CHECK-NEXT:    %[[FAST:.*]] = memref.alloc() : [[TYPE:memref<1xi32(, 2)?>]]
// DMA-NEXT:      %[[TAG:.*]] = memref.alloc() : memref<1xi32>
// CHECK-NEXT:    affine.store %[[VALUE]], %[[FAST]][0] : [[TYPE]]
// CHECK-NEXT:    affine.if #{{.*}}() {
// CHECK-NEXT:      %{{.*}} = affine.load %[[FAST]][8] : [[TYPE]]
// CHECK-NEXT:    }
// POINT-NEXT:    %[[OUT:.*]] = affine.load %[[FAST]][0] : [[TYPE]]
// POINT-NEXT:    affine.store %[[OUT]], %[[MEM]][0] : memref<1xi32>
// DMA-NEXT:      affine.dma_start %[[FAST]][%{{.*}}], %[[MEM]][%{{.*}}], %[[TAG]][%{{.*}}], %[[COUNT]] :
// DMA-NEXT:      affine.dma_wait %[[TAG]][%{{.*}}], %[[COUNT]] : memref<1xi32>
// DMA-NEXT:      memref.dealloc %[[TAG]] : memref<1xi32>
// CHECK-NEXT:    memref.dealloc %[[FAST]] : [[TYPE]]
// CHECK-NEXT:    return %[[VALUE]] : i32
func.func @unreachable_load() -> i32 {
  %m = memref.alloc() : memref<1xi32>
  %v = arith.constant 42 : i32
  affine.store %v, %m[0] : memref<1xi32>
  affine.if affine_set<() : (-1 >= 0)>() {
    %unused = affine.load %m[8] : memref<1xi32>
  }
  return %v : i32
}

// -----

// A true condition makes the else branch unreachable, including its store.
// CHECK-LABEL: func.func @unreachable_store(
// CHECK-SAME:    %[[MEM:.*]]: memref<1xi32>, %[[VALUE:.*]]: i32)
// DMA:           %[[COUNT:.*]] = arith.constant 1 : index
// CHECK:         %[[FAST:.*]] = memref.alloc() : [[TYPE:memref<1xi32(, 2)?>]]
// DMA-NEXT:      %[[TAG:.*]] = memref.alloc() : memref<1xi32>
// CHECK-NEXT:    affine.store %[[VALUE]], %[[FAST]][0] : [[TYPE]]
// CHECK-NEXT:    affine.if #{{.*}}() {
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.store %[[VALUE]], %[[FAST]][-8] : [[TYPE]]
// CHECK-NEXT:    }
// POINT-NEXT:    %[[OUT:.*]] = affine.load %[[FAST]][0] : [[TYPE]]
// POINT-NEXT:    affine.store %[[OUT]], %[[MEM]][0] : memref<1xi32>
// DMA-NEXT:      affine.dma_start %[[FAST]][%{{.*}}], %[[MEM]][%{{.*}}], %[[TAG]][%{{.*}}], %[[COUNT]] :
// DMA-NEXT:      affine.dma_wait %[[TAG]][%{{.*}}], %[[COUNT]] : memref<1xi32>
// DMA-NEXT:      memref.dealloc %[[TAG]] : memref<1xi32>
// CHECK-NEXT:    memref.dealloc %[[FAST]] : [[TYPE]]
// CHECK-NEXT:    return
func.func @unreachable_store(%m: memref<1xi32>, %v: i32) {
  affine.store %v, %m[0] : memref<1xi32>
  affine.if affine_set<() : (1 >= 0)>() {
  } else {
    affine.store %v, %m[-8] : memref<1xi32>
  }
  return
}

// -----

// Fold constant SSA operands before deciding which branch is executable.
// CHECK-LABEL: func.func @constant_then(
// CHECK-SAME:    %[[MEM:.*]]: memref<1xi32>, %[[VALUE:.*]]: i32)
// CHECK:         %[[FAST:.*]] = memref.alloc() : [[TYPE:memref<1xi32(, 2)?>]]
// DMA-NEXT:      %{{.*}} = memref.alloc() : memref<1xi32>
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 4 {
// CHECK-NEXT:      affine.if #{{.*}}()[%{{.*}}] {
// CHECK-NEXT:        affine.store %[[VALUE]], %[[FAST]][0] : [[TYPE]]
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.store %[[VALUE]], %[[FAST]][8] : [[TYPE]]
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// POINT-NEXT:    %[[OUT:.*]] = affine.load %[[FAST]][0] : [[TYPE]]
// POINT-NEXT:    affine.store %[[OUT]], %[[MEM]][0] : memref<1xi32>
// DMA-NEXT:      affine.dma_start %[[FAST]][%{{.*}}], %[[MEM]][%{{.*}}],
// CHECK:         memref.dealloc %[[FAST]] : [[TYPE]]
// CHECK-NEXT:    return
func.func @constant_then(%m: memref<1xi32>, %v: i32) {
  %zero = arith.constant 0 : index
  affine.for %i = 0 to 4 {
    affine.if affine_set<()[s0] : (s0 == 0)>()[%zero] {
      affine.store %v, %m[0] : memref<1xi32>
    } else {
      affine.store %v, %m[8] : memref<1xi32>
    }
  }
  return
}

// -----

// The else branch remains live when a constant SSA operand makes the condition
// false. Do not discard the accesses in both branches.
// CHECK-LABEL: func.func @constant_else(
// CHECK-SAME:    %[[MEM:.*]]: memref<1xi32>, %[[VALUE:.*]]: i32)
// CHECK:         %[[FAST:.*]] = memref.alloc() : [[TYPE:memref<1xi32(, 2)?>]]
// DMA-NEXT:      %{{.*}} = memref.alloc() : memref<1xi32>
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 4 {
// CHECK-NEXT:      affine.if #{{.*}}()[%{{.*}}] {
// CHECK-NEXT:        affine.store %[[VALUE]], %[[FAST]][8] : [[TYPE]]
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.store %[[VALUE]], %[[FAST]][0] : [[TYPE]]
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// POINT-NEXT:    %[[OUT:.*]] = affine.load %[[FAST]][0] : [[TYPE]]
// POINT-NEXT:    affine.store %[[OUT]], %[[MEM]][0] : memref<1xi32>
// DMA-NEXT:      affine.dma_start %[[FAST]][%{{.*}}], %[[MEM]][%{{.*}}],
// CHECK:         memref.dealloc %[[FAST]] : [[TYPE]]
// CHECK-NEXT:    return
func.func @constant_else(%m: memref<1xi32>, %v: i32) {
  %zero = arith.constant 0 : index
  affine.for %i = 0 to 4 {
    affine.if affine_set<()[s0] : (s0 - 1 == 0)>()[%zero] {
      affine.store %v, %m[8] : memref<1xi32>
    } else {
      affine.store %v, %m[0] : memref<1xi32>
    }
  }
  return
}

// -----

// Contradictory constraints prove the then branch unreachable even when the
// condition has a non-constant operand.
// CHECK-LABEL: func.func @contradictory_condition(
// CHECK-SAME:    %[[MEM:.*]]: memref<1xi32>, %[[INDEX:.*]]: index, %[[VALUE:.*]]: i32)
// CHECK:         %[[FAST:.*]] = memref.alloc() : [[TYPE:memref<1xi32(, 2)?>]]
// DMA-NEXT:      %{{.*}} = memref.alloc() : memref<1xi32>
// CHECK-NEXT:    affine.store %[[VALUE]], %[[FAST]][0] : [[TYPE]]
// CHECK-NEXT:    affine.if #{{.*}}()[%[[INDEX]]] {
// CHECK-NEXT:      affine.store %[[VALUE]], %[[FAST]][8] : [[TYPE]]
// CHECK-NEXT:    }
// POINT-NEXT:    %[[OUT:.*]] = affine.load %[[FAST]][0] : [[TYPE]]
// POINT-NEXT:    affine.store %[[OUT]], %[[MEM]][0] : memref<1xi32>
// DMA-NEXT:      affine.dma_start %[[FAST]][%{{.*}}], %[[MEM]][%{{.*}}],
// CHECK:         memref.dealloc %[[FAST]] : [[TYPE]]
// CHECK-NEXT:    return
func.func @contradictory_condition(%m: memref<1xi32>, %index: index, %v: i32) {
  affine.store %v, %m[0] : memref<1xi32>
  affine.if affine_set<()[s0] : (s0 >= 0, -s0 - 1 >= 0)>()[%index] {
    affine.store %v, %m[8] : memref<1xi32>
  }
  return
}

// -----

// Do not reject an unknown inner condition before checking an unreachable outer
// branch. The access is dead regardless of the inner condition.
// CHECK-LABEL: func.func @false_outer_unknown_inner(
// CHECK-SAME:    %[[MEM:.*]]: memref<1xi32>, %[[INDEX:.*]]: index, %[[VALUE:.*]]: i32)
// CHECK:         %[[FAST:.*]] = memref.alloc() : [[TYPE:memref<1xi32(, 2)?>]]
// DMA-NEXT:      %{{.*}} = memref.alloc() : memref<1xi32>
// CHECK-NEXT:    affine.store %[[VALUE]], %[[FAST]][0] : [[TYPE]]
// CHECK-NEXT:    affine.if #{{.*}}() {
// CHECK-NEXT:      affine.if #{{.*}}()[%[[INDEX]]] {
// CHECK-NEXT:        affine.store %[[VALUE]], %[[FAST]][8] : [[TYPE]]
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// POINT-NEXT:    %[[OUT:.*]] = affine.load %[[FAST]][0] : [[TYPE]]
// POINT-NEXT:    affine.store %[[OUT]], %[[MEM]][0] : memref<1xi32>
// DMA-NEXT:      affine.dma_start %[[FAST]][%{{.*}}], %[[MEM]][%{{.*}}],
// CHECK:         memref.dealloc %[[FAST]] : [[TYPE]]
// CHECK-NEXT:    return
func.func @false_outer_unknown_inner(%m: memref<1xi32>, %index: index, %v: i32) {
  affine.store %v, %m[0] : memref<1xi32>
  affine.if affine_set<() : (-1 >= 0)>() {
    affine.if affine_set<()[s0] : (s0 == 0)>()[%index] {
      affine.store %v, %m[8] : memref<1xi32>
    }
  }
  return
}

// -----

// An unreachable inner branch also makes the access dead independently of an
// unknown outer condition.
// CHECK-LABEL: func.func @unknown_outer_false_inner(
// CHECK-SAME:    %[[MEM:.*]]: memref<1xi32>, %[[INDEX:.*]]: index, %[[VALUE:.*]]: i32)
// CHECK:         %[[FAST:.*]] = memref.alloc() : [[TYPE:memref<1xi32(, 2)?>]]
// DMA-NEXT:      %{{.*}} = memref.alloc() : memref<1xi32>
// CHECK-NEXT:    affine.store %[[VALUE]], %[[FAST]][0] : [[TYPE]]
// CHECK-NEXT:    affine.if #{{.*}}()[%[[INDEX]]] {
// CHECK-NEXT:      affine.if #{{.*}}() {
// CHECK-NEXT:        affine.store %[[VALUE]], %[[FAST]][8] : [[TYPE]]
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// POINT-NEXT:    %[[OUT:.*]] = affine.load %[[FAST]][0] : [[TYPE]]
// POINT-NEXT:    affine.store %[[OUT]], %[[MEM]][0] : memref<1xi32>
// DMA-NEXT:      affine.dma_start %[[FAST]][%{{.*}}], %[[MEM]][%{{.*}}],
// CHECK:         memref.dealloc %[[FAST]] : [[TYPE]]
// CHECK-NEXT:    return
func.func @unknown_outer_false_inner(%m: memref<1xi32>, %index: index, %v: i32) {
  affine.store %v, %m[0] : memref<1xi32>
  affine.if affine_set<()[s0] : (s0 == 0)>()[%index] {
    affine.if affine_set<() : (-1 >= 0)>() {
      affine.store %v, %m[8] : memref<1xi32>
    }
  }
  return
}

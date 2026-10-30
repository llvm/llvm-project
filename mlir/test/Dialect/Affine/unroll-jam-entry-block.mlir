// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2 unroll-jam-all-loops}))" | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=4 unroll-jam-all-loops=true}))" | FileCheck %s --check-prefix=FOUR
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2}))" | FileCheck %s --check-prefix=DEFAULT
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2 unroll-jam-all-loops=false}))" | FileCheck %s --check-prefix=DEFAULT

// With the option enabled, setup operations must not prevent selection. No
// shared-input condition is required: the load depends on both loop IVs.
// DEFAULT-LABEL: func.func @nonloop_prefix
// DEFAULT: memref.distinct_objects
// DEFAULT-NEXT: affine.for {{.*}} = 0 to 4 {
// DEFAULT-NEXT: affine.for {{.*}} = 0 to 8 {
// DEFAULT: affine.store
// DEFAULT-NOT: affine.store
// DEFAULT: return
// CHECK-LABEL: func.func @nonloop_prefix
// CHECK: arith.constant 1 : i32
// CHECK: memref.distinct_objects
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 step 2 {
// CHECK-NEXT: affine.for {{.*}} = 0 to 8 {
// CHECK-COUNT-2: affine.store
// CHECK-NOT: affine.store
// CHECK: return
// FOUR-LABEL: func.func @nonloop_prefix
// FOUR: memref.distinct_objects
// FOUR-NEXT: affine.for {{.*}} = 0 to 8 {
// FOUR-COUNT-4: affine.store
// FOUR-NOT: affine.store
// FOUR: return
func.func @nonloop_prefix(%x: memref<4x8xi32>, %y: memref<4x8xi32>) {
  %one = arith.constant 1 : i32
  %input, %output = memref.distinct_objects %x, %y : memref<4x8xi32>, memref<4x8xi32>
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      %v = affine.load %input[%i, %j] : memref<4x8xi32>
      %sum = arith.addi %v, %one : i32
      affine.store %sum, %output[%i, %j] : memref<4x8xi32>
    }
  }
  return
}

// The initialization loop must not hide the output loop containing two
// reduction dimensions. Factor 2 partially jams the output loop; factor 4
// removes it, retaining both reduction loops and each output's strict FP chain.
// DEFAULT-LABEL: func.func @factor_two_reduction_dims
// DEFAULT: memref.alloc() : memref<4xf32>
// DEFAULT-NEXT: affine.for {{.*}} = 0 to 4 {
// DEFAULT: affine.store
// DEFAULT-NEXT: }
// DEFAULT-NEXT: affine.for {{.*}} = 0 to 4 {
// DEFAULT-NEXT: affine.for {{.*}} = 0 to 3 {
// DEFAULT-NEXT: affine.for {{.*}} = 0 to 5 {
// DEFAULT: arith.mulf
// DEFAULT-NOT: arith.mulf
// DEFAULT: return
// CHECK-LABEL: func.func @factor_two_reduction_dims(
// CHECK-SAME: [[X:%.*]]: memref<3x5xf32>, [[W:%.*]]: memref<4x3x5xf32>
// CHECK: [[ACC:%.*]] = memref.alloc() : memref<4xf32>
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 step 2 {
// CHECK-COUNT-2: affine.store
// CHECK-NEXT: }
// CHECK-NEXT: affine.for [[F:%.*]] = 0 to 4 step 2 {
// CHECK-NEXT: affine.for [[C:%.*]] = 0 to 3 {
// CHECK-NEXT: affine.for [[K:%.*]] = 0 to 5 {
// CHECK-NEXT: [[X0:%.*]] = affine.load [[X]][[[C]], [[K]]] : memref<3x5xf32>
// CHECK-NEXT: [[W0:%.*]] = affine.load [[W]][[[F]], [[C]], [[K]]] : memref<4x3x5xf32>
// CHECK-NEXT: [[A0:%.*]] = affine.load [[ACC]][[[F]]] : memref<4xf32>
// CHECK-NEXT: [[P0:%.*]] = arith.mulf [[X0]], [[W0]] : f32
// CHECK-NEXT: [[S0:%.*]] = arith.addf [[A0]], [[P0]] : f32
// CHECK-NEXT: affine.store [[S0]], [[ACC]][[[F]]] : memref<4xf32>
// CHECK-NEXT: [[F1:%.*]] = affine.apply {{.*}}([[F]])
// CHECK-NEXT: [[X1:%.*]] = affine.load [[X]][[[C]], [[K]]] : memref<3x5xf32>
// CHECK-NEXT: [[W1:%.*]] = affine.load [[W]][[[F1]], [[C]], [[K]]] : memref<4x3x5xf32>
// CHECK-NEXT: [[A1:%.*]] = affine.load [[ACC]][[[F1]]] : memref<4xf32>
// CHECK-NEXT: [[P1:%.*]] = arith.mulf [[X1]], [[W1]] : f32
// CHECK-NEXT: [[S1:%.*]] = arith.addf [[A1]], [[P1]] : f32
// CHECK-NEXT: affine.store [[S1]], [[ACC]][[[F1]]] : memref<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: return [[ACC]] : memref<4xf32>
// FOUR-LABEL: func.func @factor_two_reduction_dims
// FOUR: memref.alloc() : memref<4xf32>
// FOUR-COUNT-4: affine.store
// FOUR-NEXT: affine.for {{.*}} = 0 to 3 {
// FOUR-NEXT: affine.for {{.*}} = 0 to 5 {
// FOUR-COUNT-4: arith.mulf
// FOUR-NOT: affine.for
// FOUR: return
func.func @factor_two_reduction_dims(%x: memref<3x5xf32>, %w: memref<4x3x5xf32>) -> memref<4xf32> {
  %zero = arith.constant 0.0 : f32
  %acc = memref.alloc() : memref<4xf32>
  affine.for %i = 0 to 4 {
    affine.store %zero, %acc[%i] : memref<4xf32>
  }
  affine.for %f = 0 to 4 {
    affine.for %c = 0 to 3 {
      affine.for %k = 0 to 5 {
        %xv = affine.load %x[%c, %k] : memref<3x5xf32>
        %wv = affine.load %w[%f, %c, %k] : memref<4x3x5xf32>
        %old = affine.load %acc[%f] : memref<4xf32>
        %product = arith.mulf %xv, %wv : f32
        %sum = arith.addf %old, %product : f32
        affine.store %sum, %acc[%f] : memref<4xf32>
      }
    }
  }
  return %acc : memref<4xf32>
}

// Select both entry-block nests, including a loop after an intervening op.
// Without the option, only the first nest is selected.
// Nested loops must not be selected again after their outer loop is jammed.
// DEFAULT-LABEL: func.func @separate_nests
// DEFAULT: affine.for {{.*}} = 0 to 4 step 2 {
// DEFAULT-NEXT: affine.for {{.*}} = 0 to 8 {
// DEFAULT-COUNT-2: affine.store
// DEFAULT: arith.addi
// DEFAULT-NEXT: affine.for {{.*}} = 0 to 4 {
// DEFAULT-NEXT: affine.for {{.*}} = 0 to 8 {
// DEFAULT: affine.store
// DEFAULT-NOT: affine.store
// DEFAULT: return
// CHECK-LABEL: func.func @separate_nests
// CHECK: affine.for {{.*}} = 0 to 4 step 2 {
// CHECK-NEXT: affine.for {{.*}} = 0 to 8 {
// CHECK-COUNT-2: affine.store
// CHECK: arith.addi
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 step 2 {
// CHECK-NEXT: affine.for {{.*}} = 0 to 8 {
// CHECK-COUNT-2: affine.store
// CHECK-NOT: affine.store
// CHECK: return
// FOUR-LABEL: func.func @separate_nests
// FOUR: affine.for {{.*}} = 0 to 8 {
// FOUR-COUNT-4: affine.store
// FOUR: arith.addi
// FOUR-NEXT: affine.for {{.*}} = 0 to 8 {
// FOUR-COUNT-4: affine.store
// FOUR-NOT: affine.store
// FOUR: return
func.func @separate_nests(%a: memref<4x8xi32>, %b: memref<4x8xi32>, %value: i32) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      affine.store %value, %a[%i, %j] : memref<4x8xi32>
    }
  }
  %next = arith.addi %value, %value : i32
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      affine.store %next, %b[%i, %j] : memref<4x8xi32>
    }
  }
  return
}

// Promoting a single-iteration cleanup exposes another entry-block loop.
// Neither that loop nor the main nest's inner loop should be revisited.
// CHECK-LABEL: func.func @cleanup_not_revisited
// CHECK: affine.for {{.*}} = 0 to 4 step 2 {
// CHECK-NEXT: affine.for {{.*}} = 0 to 8 {
// CHECK-COUNT-2: affine.store
// CHECK: affine.for {{.*}} = 0 to 8 {
// CHECK: affine.store
// CHECK-NOT: affine.store
// CHECK: return
// FOUR-LABEL: func.func @cleanup_not_revisited
// FOUR: affine.for {{.*}} = 0 to 8 {
// FOUR-COUNT-4: affine.store
// FOUR: affine.for {{.*}} = 0 to 8 {
// FOUR: affine.store
// FOUR-NOT: affine.store
// FOUR: return
func.func @cleanup_not_revisited(%a: memref<5x8xi32>, %value: i32) {
  affine.for %i = 0 to 5 {
    affine.for %j = 0 to 8 {
      affine.store %value, %a[%i, %j] : memref<5x8xi32>
    }
  }
  return
}

// DEFAULT-LABEL: func.func @no_loop
// DEFAULT-NEXT: return
// CHECK-LABEL: func.func @no_loop
// CHECK-NEXT: return
// FOUR-LABEL: func.func @no_loop
// FOUR-NEXT: return
func.func @no_loop() {
  return
}

func.func private @external()

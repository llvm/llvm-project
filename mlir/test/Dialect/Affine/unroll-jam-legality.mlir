// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2 unroll-jam-all-loops}))" | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2}))" | FileCheck %s --check-prefix=LEGACY

// CHECK-DAG: [[$PLUS_ONE:#map[0-9]*]] = affine_map<(d0) -> (d0 + 1)>

// The first operation's loop keeps the legacy behavior, even when the option
// is enabled. The new independence proof applies only to additional targets.
// CHECK-LABEL: func.func @carried_memory_dependence
// CHECK-NEXT: affine.for {{.*}} = 1 to 5 step 2 {
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK-COUNT-2: affine.store
// CHECK: return
// LEGACY-LABEL: func.func @carried_memory_dependence
// LEGACY-NEXT: affine.for {{.*}} = 1 to 5 step 2 {
// LEGACY-NEXT: affine.for {{.*}} = 0 to 4 {
// LEGACY-COUNT-2: affine.store
// LEGACY: return
func.func @carried_memory_dependence(%a: memref<5x5xi32>) {
  affine.for %i = 1 to 5 {
    affine.for %j = 0 to 4 {
      %v = affine.load %a[%i - 1, %j + 1] : memref<5x5xi32>
      affine.store %v, %a[%i, %j] : memref<5x5xi32>
    }
  }
  return
}

// The existing target is retained even when its distinct memref arguments
// cannot be proven NoAlias. Additional targets must establish that proof.
// CHECK-LABEL: func.func @may_alias
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 step 2 {
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK-COUNT-2: affine.store
// CHECK: return
// LEGACY-LABEL: func.func @may_alias
// LEGACY-NEXT: affine.for {{.*}} = 0 to 4 step 2 {
// LEGACY-COUNT-2: affine.store
// LEGACY: return
func.func @may_alias(%input: memref<5xi32>, %output: memref<4xi32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %v = affine.load %input[%j + 1] : memref<5xi32>
      affine.store %v, %output[%i] : memref<4xi32>
    }
  }
  return
}

// The first nest is the legacy target. The second nest is an additional
// target: jamming would move the next row's read before the previous row's
// write at j + 1, so it must remain unchanged with the option enabled.
// CHECK-LABEL: func.func @additional_carried_memory_dependence
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 step 2 {
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK-COUNT-2: affine.store
// CHECK: affine.for {{.*}} = 1 to 5 {
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK: affine.store
// CHECK-NOT: affine.store
// CHECK: return
// LEGACY-LABEL: func.func @additional_carried_memory_dependence
// LEGACY-NEXT: affine.for {{.*}} = 0 to 4 step 2 {
// LEGACY-NEXT: affine.for {{.*}} = 0 to 4 {
// LEGACY-COUNT-2: affine.store
// LEGACY: affine.for {{.*}} = 1 to 5 {
// LEGACY-NEXT: affine.for {{.*}} = 0 to 4 {
// LEGACY: affine.store
// LEGACY-NOT: affine.store
// LEGACY: return
func.func @additional_carried_memory_dependence(%a: memref<5x5xi32>, %setup: memref<4x4xi32>, %value: i32) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      affine.store %value, %setup[%i, %j] : memref<4x4xi32>
    }
  }
  affine.for %i = 1 to 5 {
    affine.for %j = 0 to 4 {
      %v = affine.load %a[%i - 1, %j + 1] : memref<5x5xi32>
      affine.store %v, %a[%i, %j] : memref<5x5xi32>
    }
  }
  return
}

// A loop after setup is also an additional target. Distinct SSA values do
// not prove NoAlias, so this loop must remain unchanged when the option is on.
// CHECK-LABEL: func.func @additional_may_alias
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK: affine.store
// CHECK-NOT: affine.store
// CHECK: return
func.func @additional_may_alias(%input: memref<5xi32>, %output: memref<4xi32>) {
  %one = arith.constant 1 : i32
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %v = affine.load %input[%j + 1] : memref<5xi32>
      %next = arith.addi %v, %one : i32
      affine.store %next, %output[%i] : memref<4xi32>
    }
  }
  return
}

// Views of one allocation can alias even when their SSA values differ.
// CHECK-LABEL: func.func @aliasing_cast
// CHECK: memref.cast
// CHECK-NEXT: affine.for {{.*}} = 1 to 5 {
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK: affine.store
// CHECK-NOT: affine.store
// CHECK: return
func.func @aliasing_cast(%a: memref<5x5xi32>) {
  %view = memref.cast %a : memref<5x5xi32> to memref<?x?xi32>
  affine.for %i = 1 to 5 {
    affine.for %j = 0 to 4 {
      %v = affine.load %a[%i - 1, %j + 1] : memref<5x5xi32>
      affine.store %v, %view[%i, %j] : memref<?x?xi32>
    }
  }
  return
}

// Outer iter_args carry an SSA dependence. In particular, do not reassociate
// a strict floating-point recurrence across the iterations being jammed.
// CHECK-LABEL: func.func @outer_iter_args
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: {{.*}} = affine.for {{.*}} = 0 to 4 iter_args
// CHECK: arith.addf
// CHECK-NOT: arith.addf
// CHECK: return
func.func @outer_iter_args(%a: memref<4x4xf32>, %delta: f32) -> f32 {
  %initial = arith.constant 1.0 : f32
  %r = affine.for %i = 0 to 4 iter_args(%acc = %initial) -> f32 {
    affine.for %j = 0 to 4 {
      affine.store %acc, %a[%i, %j] : memref<4x4xf32>
    }
    %next = arith.addf %acc, %delta : f32
    affine.yield %next : f32
  }
  return %r : f32
}

// Calls with unknown effects prevent the independence proof.
// CHECK-LABEL: func.func @unknown_effects
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK: call @effect
// CHECK-NOT: call @effect
// CHECK: return
func.func @unknown_effects(%a: memref<4x4xi32>, %value: i32) {
  %one = arith.constant 1 : i32
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      func.call @effect(%i, %j) : (index, index) -> ()
      %next = arith.addi %value, %one : i32
      affine.store %next, %a[%i, %j] : memref<4x4xi32>
    }
  }
  return
}
func.func private @effect(index, index)

// Memory-effect freedom alone is insufficient for arbitrary operations.
// Keep this first check conservative for non-speculatable computations.
// CHECK-LABEL: func.func @non_speculatable
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK: arith.divsi
// CHECK-NOT: arith.divsi
// CHECK: return
func.func @non_speculatable(%a: memref<4x4xi32>, %value: i32, %divisor: i32) {
  %one = arith.constant 1 : i32
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %v = arith.divsi %value, %divisor : i32
      %next = arith.addi %v, %one : i32
      affine.store %next, %a[%i, %j] : memref<4x4xi32>
    }
  }
  return
}

// Do not infer execution properties for regions other than affine loops,
// even if all their nested operations are memory-effect-free.
// CHECK-LABEL: func.func @unsupported_region
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK: scf.execute_region
// CHECK-NOT: scf.execute_region
// CHECK: return
func.func @unsupported_region(%a: memref<4x4xi32>, %value: i32) {
  %one = arith.constant 1 : i32
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %v = scf.execute_region -> i32 {
        %sum = arith.addi %value, %one : i32
        scf.yield %sum : i32
      }
      affine.store %v, %a[%i, %j] : memref<4x4xi32>
    }
  }
  return
}

// The varying view offset cancels i: every iteration accesses row 4. Modeling
// the view as a fixed base would incorrectly infer independence from its i index.
// CHECK-LABEL: func.func @varying_memref_base
// CHECK-NEXT: arith.constant 4 : index
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK: memref.subview
// CHECK: affine.store
// CHECK-NOT: affine.store
// CHECK: return
func.func @varying_memref_base(%a: memref<8x4xi32>) {
  %four = arith.constant 4 : index
  affine.for %i = 0 to 4 {
    %offset = affine.apply affine_map<(d0)[s0] -> (s0 - d0)>(%i)[%four]
    %view = memref.subview %a[%offset, 0] [4, 4] [1, 1] : memref<8x4xi32> to memref<4x4xi32, strided<[4, 1], offset: ?>>
    affine.for %j = 0 to 3 {
      %v = affine.load %view[%i, %j + 1] : memref<4x4xi32, strided<[4, 1], offset: ?>>
      affine.store %v, %view[%i, %j] : memref<4x4xi32, strided<[4, 1], offset: ?>>
    }
  }
  return
}

// A semi-affine access outside the dependence analysis's supported relations
// must be skipped when the analysis fails to prove independence.
// CHECK-LABEL: func.func @unanalyzable_access
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
// CHECK: affine.store
// CHECK-NOT: affine.store
// CHECK: return
func.func @unanalyzable_access(%a: memref<?x4xi32>, %value: i32, %stride: index) {
  %one = arith.constant 1 : i32
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %next = arith.addi %value, %one : i32
      affine.store %next, %a[%i * symbol(%stride), %j] : memref<?x4xi32>
    }
  }
  return
}

// Inner iter_args do not carry a dependence across the selected outer loop.
// Keep each output's nonzero initial value and strict FP chain separate.
// CHECK-LABEL: func.func @inner_iter_args(
// CHECK-SAME: [[X:%.*]]: memref<5xf32>, [[A:%.*]]: memref<4xf32>, [[INIT:%.*]]: f32
// CHECK: memref.distinct_objects
// CHECK-NEXT: affine.for [[I:%.*]] = 0 to 4 step 2 {
// CHECK-NEXT: [[R:%.*]]:2 = affine.for {{.*}} = 0 to 5 iter_args([[ACC0:%.*]] = [[INIT]], [[ACC1:%.*]] = [[INIT]]) -> (f32, f32) {
// CHECK: [[SUM0:%.*]] = arith.addf [[ACC0]], {{.*}} : f32
// CHECK: [[SUM1:%.*]] = arith.addf [[ACC1]], {{.*}} : f32
// CHECK-NEXT: affine.yield [[SUM0]], [[SUM1]] : f32, f32
// CHECK-NEXT: }
// CHECK-NEXT: affine.store [[R]]#0, [[OUT:%.*]][[[I]]] : memref<4xf32>
// CHECK-NEXT: [[I1:%.*]] = affine.apply [[$PLUS_ONE]]([[I]])
// CHECK-NEXT: affine.store [[R]]#1, [[OUT]][[[I1]]] : memref<4xf32>
// CHECK-NOT: arith.addf
// CHECK: return
func.func @inner_iter_args(%x: memref<5xf32>, %a: memref<4xf32>, %initial: f32) {
  %input, %output = memref.distinct_objects %x, %a : memref<5xf32>, memref<4xf32>
  affine.for %i = 0 to 4 {
    %r = affine.for %k = 0 to 5 iter_args(%acc = %initial) -> f32 {
      %v = affine.load %input[%k] : memref<5xf32>
      %next = arith.addf %acc, %v : f32
      affine.yield %next : f32
    }
    affine.store %r, %output[%i] : memref<4xf32>
  }
  return
}

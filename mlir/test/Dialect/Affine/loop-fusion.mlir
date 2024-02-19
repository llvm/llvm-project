// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion))' -split-input-file | FileCheck %s

// Part II of fusion tests in  mlir/test/Transforms/loop-fusion=2.mlir.
// Part III of fusion tests in mlir/test/Transforms/loop-fusion-3.mlir
// Part IV of fusion tests in mlir/test/Transforms/loop-fusion-4.mlir

// TODO: Add more tests:
// *) Add nested fusion test cases when non-constant loop bound support is
//    added to iteration domain in dependence check.
// *) Add a test w/ floordiv/ceildiv/mod when supported in dependence check.
// *) Add tests which check fused computation slice indexing and loop bounds.
// TODO: Test clean up: move memref allocs to func args.

// -----

// CHECK-LABEL: func @should_fuse_raw_dep_for_locality() {
func.func @should_fuse_raw_dep_for_locality() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_reduction_to_pointwise() {
func.func @should_fuse_reduction_to_pointwise() {
  %a = memref.alloc() : memref<10x10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %v0 = affine.load %b[%i0] : memref<10xf32>
      %v1 = affine.load %a[%i0, %i1] : memref<10x10xf32>
      %v3 = arith.addf %v0, %v1 : f32
      affine.store %v3, %b[%i0] : memref<10xf32>
    }
  }
  affine.for %i2 = 0 to 10 {
    %v4 = affine.load %b[%i2] : memref<10xf32>
    affine.store %v4, %c[%i2] : memref<10xf32>
  }

  // Should fuse in entire inner loop on %i1 from source loop nest, as %i1
  // is not used in the access function of the store/load on %b.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:      affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
  // CHECK-NEXT:      arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-DAG: [[$MAP_SHIFT_MINUS_ONE_R1:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 - 1)>
// CHECK-DAG: [[$MAP_SHIFT_D0_BY_ONE:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (d0 + 1)>
// CHECK-DAG: [[$MAP_SHIFT_D1_BY_ONE:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (d1 + 1)>

// CHECK-LABEL: func @should_fuse_loop_nests_with_shifts() {
func.func @should_fuse_loop_nests_with_shifts() {
  %a = memref.alloc() : memref<10x10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 9 {
    affine.for %i1 = 0 to 9 {
      affine.store %cf7, %a[%i0 + 1, %i1 + 1] : memref<10x10xf32>
    }
  }
  affine.for %i2 = 1 to 10 {
    affine.for %i3 = 1 to 10 {
      %v0 = affine.load %a[%i2, %i3] : memref<10x10xf32>
    }
  }

  // Source slice affine apply sequence:
  // *) First two affine apply's map from the dst to src iteration space.
  // *) Third affine apply is access function around src store.
  // *) Fourth affine apply shifts the stores access function by '-1', because
  //    of the offset induced by reducing the memref shape from 10x10 to 9x9.
  // *) Fifth affine apply shifts the loads access function by '-1', because
  //    of the offset induced by reducing the memref shape from 10x10 to 9x9.
  // NOTE: Should create a private memref with reduced shape 9x9xf32.
  // CHECK:      affine.for %{{.*}} = 1 to 10 {
  // CHECK-NEXT:   affine.for %{{.*}} = 1 to 10 {
  // CHECK-NEXT:     %[[I:.*]] = affine.apply [[$MAP_SHIFT_MINUS_ONE_R1]](%{{.*}})
  // CHECK-NEXT:     %[[J:.*]] = affine.apply [[$MAP_SHIFT_MINUS_ONE_R1]](%{{.*}})
  // CHECK-NEXT:     affine.apply [[$MAP_SHIFT_D0_BY_ONE]](%[[I]], %[[J]])
  // CHECK-NEXT:     affine.apply [[$MAP_SHIFT_D1_BY_ONE]](%[[I]], %[[J]])
  // CHECK-NEXT:     affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:     affine.load %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_loop_nest() {
func.func @should_fuse_loop_nest() {
  %a = memref.alloc() : memref<10x10xf32>
  %b = memref.alloc() : memref<10x10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.store %cf7, %a[%i0, %i1] : memref<10x10xf32>
    }
  }
  affine.for %i2 = 0 to 10 {
    affine.for %i3 = 0 to 10 {
      %v0 = affine.load %a[%i3, %i2] : memref<10x10xf32>
      affine.store %v0, %b[%i2, %i3] : memref<10x10xf32>
    }
  }
  affine.for %i4 = 0 to 10 {
    affine.for %i5 = 0 to 10 {
      %v1 = affine.load %b[%i4, %i5] : memref<10x10xf32>
    }
  }
  // Expecting private memref for '%a' first, then private memref for '%b'.
  // CHECK-DAG:  [[NEWA:%[0-9a-zA-Z_]+]] = memref.alloc() : memref<1x1xf32>
  // CHECK-DAG:  [[NEWB:%[0-9a-zA-Z_]+]] = memref.alloc() : memref<1x1xf32>
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:     affine.store %{{.*}}, [[NEWA]][0, 0] : memref<1x1xf32>
  // CHECK-NEXT:     affine.load [[NEWA]][0, 0] : memref<1x1xf32>
  // CHECK-NEXT:     affine.store %{{.*}}, [[NEWB]][0, 0] : memref<1x1xf32>
  // CHECK-NEXT:     affine.load [[NEWB]][0, 0] : memref<1x1xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_across_intermediate_loop_with_no_deps() {
func.func @should_fuse_across_intermediate_loop_with_no_deps() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %a[%i0] : memref<10xf32>
    affine.store %v0, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %c[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v1 = affine.load %b[%i2] : memref<10xf32>
  }

  // Should fuse first loop (past second loop with no dependences) into third.
  // Note that fusion creates a private memref '%2' for the fused loop nest.
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_all_loops() {
func.func @should_fuse_all_loops() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  // Set up flow dependences from first and second loops to third.
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %a[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %b[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v0 = affine.load %a[%i2] : memref<10xf32>
    %v1 = affine.load %b[%i2] : memref<10xf32>
  }

  // Should fuse first and second loops into third.
  // Expecting private memref for '%a' first, then private memref for '%b'.
  // CHECK-DAG: [[NEWA:%[0-9a-zA-Z_]+]] = memref.alloc() : memref<1xf32>
  // CHECK-DAG: [[NEWB:%[0-9a-zA-Z_]+]] = memref.alloc() : memref<1xf32>
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, [[NEWA]][0] : memref<1xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, [[NEWB]][0] : memref<1xf32>
  // CHECK-NEXT:   affine.load [[NEWA]][0] : memref<1xf32>
  // CHECK-NEXT:   affine.load [[NEWB]][0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_first_and_second_loops() {
func.func @should_fuse_first_and_second_loops() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %a[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %a[%i1] : memref<10xf32>
    affine.store %cf7, %b[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v1 = affine.load %c[%i2] : memref<10xf32>
  }

  // Should fuse first loop into the second (last loop should not be fused).
  // Should create private memref '%2' for fused scf.
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return

  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_would_create_cycle() {
func.func @should_not_fuse_would_create_cycle() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = arith.constant 7.0 : f32

  // Set up the following dependences:
  // 1) loop0 -> loop1 on memref '%{{.*}}'
  // 2) loop0 -> loop2 on memref '%{{.*}}'
  // 3) loop1 -> loop2 on memref '%{{.*}}'
  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %a[%i0] : memref<10xf32>
    affine.store %cf7, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %a[%i1] : memref<10xf32>
    %v1 = affine.load %c[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v2 = affine.load %b[%i2] : memref<10xf32>
    affine.store %cf7, %c[%i2] : memref<10xf32>
  }
  // Should not fuse: fusing loop first loop into last would create a cycle.
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_producer_consumer() {
func.func @should_fuse_producer_consumer() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %m[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v1 = affine.load %m[%i2] : memref<10xf32>
  }
  // Fusing loop %i0 to %i2 would violate the WAW dependence between %i0 and
  // %i1, but OK to fuse %i1 into %i2.
  // TODO: When the fusion pass is run to a fixed-point, it should
  // fuse all three of these loop nests.
  // CHECK:      memref.alloc() : memref<1xf32>
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_and_move_to_preserve_war_dep() {
func.func @should_fuse_and_move_to_preserve_war_dep() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %a[%i0] : memref<10xf32>
    affine.store %v0, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %a[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v1 = affine.load %b[%i2] : memref<10xf32>
  }
  // Loops '%i1' and '%i2' have no dependences. We can fuse a slice of '%i0'
  // into '%i2' if we move the fused loop nest before '%i1', which preserves
  // the WAR dependence from load '%a' in '%i0' to the store '%a' in loop '%i1'.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_if_top_level_access() {
func.func @should_fuse_if_top_level_access() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }

  %c0 = arith.constant 4 : index
  %v1 = affine.load %m[%c0] : memref<10xf32>
  // Top-level load to '%m' should prevent creating a private memref but
  // loop nests should be fused and '%i0' should be removed.
  // CHECK:      %[[m:.*]] = memref.alloc() : memref<10xf32>
  // CHECK-NOT:  memref.alloc

  // CHECK:      affine.for %[[i1:.*]] = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %[[m]][%[[i1]]] : memref<10xf32>
  // CHECK-NEXT:   affine.load %[[m]][%[[i1]]] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      affine.load %[[m]][%{{.*}}] : memref<10xf32>
  return
}

// -----

// CHECK-LABEL: func @should_fuse_but_not_remove_src() {
func.func @should_fuse_but_not_remove_src() {
  %m = memref.alloc() : memref<100xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 100 {
    affine.store %cf7, %m[%i0] : memref<100xf32>
  }
  affine.for %i1 = 0 to 17 {
    %v0 = affine.load %m[%i1] : memref<100xf32>
  }
  %v1 = affine.load %m[99] : memref<100xf32>

  // Loop '%i0' and '%i1' should be fused but '%i0' shouldn't be removed to
  // preserve the dependence with the top-level access.
  // CHECK:      affine.for %{{.*}} = 0 to 100 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<100xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 17 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.load %{{.*}}[99] : memref<100xf32>
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_no_top_level_access() {
func.func @should_fuse_no_top_level_access() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

#set0 = affine_set<(d0) : (1 == 0)>

// CHECK-LABEL: func @should_not_fuse_if_op_at_top_level() {
func.func @should_not_fuse_if_op_at_top_level() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }
  %c0 = arith.constant 4 : index
  affine.if #set0(%c0) {
  }
  // Top-level IfOp should prevent fusion.
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  return
}

// -----

#set0 = affine_set<(d0) : (1 == 0)>

// CHECK-LABEL: func @should_not_fuse_if_op_in_loop_nest() {
func.func @should_not_fuse_if_op_in_loop_nest() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %c4 = arith.constant 4 : index

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.if #set0(%c4) {
    }
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }

  // IfOp in ForOp should prevent fusion.
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.if #set(%{{.*}}) {
  // CHECK-NEXT:   }
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  return
}

// -----

#set = affine_set<(d0) : (d0 - 1 >= 0)>

// CHECK-LABEL: func @should_fuse_if_op_in_loop_nest_not_sandwiched() -> memref<10xf32> {
func.func @should_fuse_if_op_in_loop_nest_not_sandwiched() -> memref<10xf32> {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %a[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %a[%i1] : memref<10xf32>
    affine.store %v0, %b[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    affine.if #set(%i2) {
      %v0 = affine.load %b[%i2] : memref<10xf32>
    }
  }

  // IfOp in ForOp should not prevent fusion if it does not in between the
  // source and dest ForOp ops.

  // CHECK:      affine.for
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.store
  // CHECK:      affine.for
  // CHECK-NEXT:   affine.if
  // CHECK-NEXT:     affine.load
  // CHECK-NOT:  affine.for
  // CHECK:      return

  return %a : memref<10xf32>
}

// -----

#set = affine_set<(d0) : (d0 - 1 >= 0)>

// CHECK-LABEL: func @should_not_fuse_if_op_in_loop_nest_between_src_and_dest() -> memref<10xf32> {
func.func @should_not_fuse_if_op_in_loop_nest_between_src_and_dest() -> memref<10xf32> {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %a[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.if #set(%i1) {
      affine.store %cf7, %a[%i1] : memref<10xf32>
    }
  }
  affine.for %i3 = 0 to 10 {
    %v0 = affine.load %a[%i3] : memref<10xf32>
    affine.store %v0, %b[%i3] : memref<10xf32>
  }
  return %b : memref<10xf32>

  // IfOp in ForOp which modifies the memref should prevent fusion if it is in
  // between the source and dest ForOp.

  // CHECK:      affine.for
  // CHECK-NEXT:   affine.store
  // CHECK:      affine.for
  // CHECK-NEXT:   affine.if
  // CHECK-NEXT:     affine.store
  // CHECK:      affine.for
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.store
  // CHECK:      return
}

// -----

// CHECK-LABEL: func @permute_and_fuse() {
func.func @permute_and_fuse() {
  %m = memref.alloc() : memref<10x20x30xf32>

  %cf7 = arith.constant 7.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 20 {
      affine.for %i2 = 0 to 30 {
        affine.store %cf7, %m[%i0, %i1, %i2] : memref<10x20x30xf32>
      }
    }
  }
  affine.for %i3 = 0 to 30 {
    affine.for %i4 = 0 to 10 {
      affine.for %i5 = 0 to 20 {
        %v0 = affine.load %m[%i4, %i5, %i3] : memref<10x20x30xf32>
        "foo"(%v0) : (f32) -> ()
      }
    }
  }
// CHECK:       affine.for %{{.*}} = 0 to 30 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:      affine.for %{{.*}} = 0 to 20 {
// CHECK-NEXT:        affine.store %{{.*}}, %{{.*}}[0, 0, 0] : memref<1x1x1xf32>
// CHECK-NEXT:        affine.load %{{.*}}[0, 0, 0] : memref<1x1x1xf32>
// CHECK-NEXT:        "foo"(%{{.*}}) : (f32) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return

  return
}

// -----

// CHECK-DAG: [[$MAP0:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>
// CHECK-DAG: [[$MAP1:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 floordiv 4)>
// CHECK-DAG: [[$MAP2:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 mod 4)>

// Reshape from a 64 x f32 to 16 x 4 x f32.
// CHECK-LABEL: func @fuse_reshape_64_16_4
func.func @fuse_reshape_64_16_4(%in : memref<64xf32>) {
  %out = memref.alloc() : memref<16x4xf32>

  affine.for %i0 = 0 to 64 {
    %v = affine.load %in[%i0] : memref<64xf32>
    affine.store %v, %out[%i0 floordiv 4, %i0 mod 4] : memref<16x4xf32>
  }

  affine.for %i1 = 0 to 16 {
    affine.for %i2 = 0 to 4 {
      %w = affine.load %out[%i1, %i2] : memref<16x4xf32>
      "foo"(%w) : (f32) -> ()
    }
  }
  return
  // CHECK:      affine.for %{{.*}} =
  // CHECK-NEXT:   affine.for %{{.*}} =
  // CHECK-NOT:    for
  // CHECK:        }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
}

// -----
// CHECK-DAG: [[$MAP0:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 floordiv 4)>
// CHECK-DAG: [[$MAP1:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 mod 4)>
// CHECK-DAG: [[$MAP2:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>

// Reshape a 16x4xf32 to 64xf32.
// CHECK-LABEL: func @fuse_reshape_16_4_64
func.func @fuse_reshape_16_4_64() {
  %in = memref.alloc() : memref<16x4xf32>
  %out = memref.alloc() : memref<64xf32>

  affine.for %i0 = 0 to 16 {
    affine.for %i1 = 0 to 4 {
      %v = affine.load %in[%i0, %i1] : memref<16x4xf32>
      affine.store %v, %out[4*%i0 + %i1] : memref<64xf32>
    }
  }

  affine.for %i2 = 0 to 64 {
    %w = affine.load %out[%i2] : memref<64xf32>
    "foo"(%w) : (f32) -> ()
  }
// CHECK:       affine.for %{{.*}} = 0 to 64 {
// CHECK-NEXT:    affine.apply [[$MAP0]](%{{.*}})
// CHECK-NEXT:    affine.apply [[$MAP1]](%{{.*}})
// CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<16x4xf32>
// CHECK-NEXT:    affine.apply [[$MAP2]](%{{.*}}, %{{.*}})
// CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
// CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
// CHECK-NEXT:    "foo"(%{{.*}}) : (f32) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  return
  return
}


// -----

// All three loop nests below (6-d one, 2-d one, 2-d one is fused into a single
// 2-d loop nest).
func.func @R6_to_R2_reshape_square() -> memref<64x9xi32> {
  %in = memref.alloc() : memref<2x2x3x3x16x1xi32>
  %out = memref.alloc() : memref<64x9xi32>
  %live_out = memref.alloc() : memref<64x9xi32>

  // Initialize input.
  affine.for %i0 = 0 to 2 {
    affine.for %i1 = 0 to 2 {
      affine.for %i2 = 0 to 3 {
        affine.for %i3 = 0 to 3 {
          affine.for %i4 = 0 to 16 {
            affine.for %i5 = 0 to 1 {
              %val = "foo"(%i0, %i1, %i2, %i3, %i4, %i5) : (index, index, index, index, index, index) -> i32
              affine.store %val, %in[%i0, %i1, %i2, %i3, %i4, %i5] : memref<2x2x3x3x16x1xi32>
            }
          }
        }
      }
    }
  }

  affine.for %ii = 0 to 64 {
    affine.for %jj = 0 to 9 {
      // Convert output coordinates to linear index.
      %a0 = affine.apply affine_map<(d0, d1) -> (d0 * 9 + d1)> (%ii, %jj)
      %0 = affine.apply affine_map<(d0) -> (d0 floordiv (2 * 3 * 3 * 16 * 1))>(%a0)
      %1 = affine.apply affine_map<(d0) -> ((d0 mod 288) floordiv (3 * 3 * 16 * 1))>(%a0)
      %2 = affine.apply affine_map<(d0) -> (((d0 mod 288) mod 144) floordiv (3 * 16 * 1))>(%a0)
      %3 = affine.apply affine_map<(d0) -> ((((d0 mod 288) mod 144) mod 48) floordiv (16 * 1))>(%a0)
      %4 = affine.apply affine_map<(d0) -> ((((d0 mod 288) mod 144) mod 48) mod 16)>(%a0)
      %5 = affine.apply affine_map<(d0) -> (((((d0 mod 144) mod 144) mod 48) mod 16) mod 1)>(%a0)
      %v = affine.load %in[%0, %1, %2, %3, %4, %5] : memref<2x2x3x3x16x1xi32>
      affine.store %v, %out[%ii, %jj] : memref<64x9xi32>
    }
  }

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 9 {
      %a = affine.load %out[%i, %j] : memref<64x9xi32>
      %b = arith.muli %a, %a : i32
      affine.store %b, %live_out[%i, %j] : memref<64x9xi32>
    }
  }
  return %live_out : memref<64x9xi32>
}
// Everything above is fused to a single 2-d loop nest, and the 6-d tensor %in
// is eliminated if -memref-dataflow-opt is also supplied.
//
// CHECK-DAG: [[$MAP0:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> ((d0 * 9 + d1) floordiv 288)>
// CHECK-DAG: [[$MAP1:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (((d0 * 9 + d1) mod 288) floordiv 144)>
// CHECK-DAG: [[$MAP2:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (((d0 * 9 + d1) mod 144) floordiv 48)>
// CHECK-DAG: [[$MAP3:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (((d0 * 9 + d1) mod 48) floordiv 16)>
// CHECK-DAG: [[$MAP4:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> ((d0 * 9 + d1) mod 16)>
// CHECK-DAG: [[$MAP11:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (d0 * 9 + d1)>
// CHECK-DAG: [[$MAP12:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 floordiv 288)>
// CHECK-DAG: [[$MAP13:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> ((d0 mod 288) floordiv 144)>
// CHECK-DAG: [[$MAP14:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> ((d0 mod 144) floordiv 48)>
// CHECK-DAG: [[$MAP15:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> ((d0 mod 48) floordiv 16)>
// CHECK-DAG: [[$MAP16:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 mod 16)>
// CHECK-DAG: [[$MAP17:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (0)>

//
// CHECK-LABEL: func @R6_to_R2_reshape
// CHECK:       memref.alloc() : memref<1x2x3x3x16x1xi32>
// CHECK:       memref.alloc() : memref<1x1xi32>
// CHECK:       memref.alloc() : memref<64x9xi32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 64 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 9 {
// CHECK-NEXT:      affine.apply [[$MAP0]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP1]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP2]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP3]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP4]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      "foo"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (index, index, index, index, index, index) -> i32
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0, ((%{{.*}} * 9 + %{{.*}}) mod 288) floordiv 144, ((%{{.*}} * 9 + %{{.*}}) mod 144) floordiv 48, ((%{{.*}} * 9 + %{{.*}}) mod 48) floordiv 16, (%{{.*}} * 9 + %{{.*}}) mod 16, 0] : memref<1x2x3x3x16x1xi32>
// CHECK-NEXT:      affine.apply [[$MAP11]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP12]](%{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP13]](%{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP14]](%{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP15]](%{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP16]](%{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP17]](%{{.*}})
// CHECK-NEXT:      affine.load %{{.*}}[0, ((%{{.*}} * 9 + %{{.*}}) mod 288) floordiv 144, ((%{{.*}} * 9 + %{{.*}}) mod 144) floordiv 48, ((%{{.*}} * 9 + %{{.*}}) mod 48) floordiv 16, (%{{.*}} * 9 + %{{.*}}) mod 16, 0] : memref<1x2x3x3x16x1xi32>
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xi32>
// CHECK-NEXT:      affine.load %{{.*}}[0, 0] : memref<1x1xi32>
// CHECK-NEXT:      arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<64x9xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return %{{.*}} : memref<64x9xi32>

// -----

// CHECK-LABEL: func @fuse_symbolic_bounds
func.func @fuse_symbolic_bounds(%M : index, %N : index) {
  %N_plus_5 = affine.apply affine_map<(d0) -> (d0 + 5)>(%N)
  %m = memref.alloc(%M, %N_plus_5) : memref<? x ? x f32>

  %c0 = arith.constant 0.0 : f32
  %s = arith.constant 5 : index

  affine.for %i0 = 0 to %M {
    affine.for %i1 = 0 to affine_map<(d0) -> (d0 + 5)> (%N) {
      affine.store %c0, %m[%i0, %i1] : memref<? x ? x f32>
    }
  }

  affine.for %i2 = 0 to %M {
    affine.for %i3 = 0 to %N {
      %v = affine.load %m[%i2, %i3 + symbol(%s)] : memref<? x ? x f32>
    }
  }

  return
}

// -----

// CHECK-LABEL: func @should_fuse_reduction_at_depth_of_one
func.func @should_fuse_reduction_at_depth_of_one() {
  %a = memref.alloc() : memref<10x100xf32>
  %b = memref.alloc() : memref<10xf32>

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 100 {
      %v0 = affine.load %b[%i0] : memref<10xf32>
      %v1 = affine.load %a[%i0, %i1] : memref<10x100xf32>
      %v2 = "maxf"(%v0, %v1) : (f32, f32) -> f32
      affine.store %v2, %b[%i0] : memref<10xf32>
    }
  }
  affine.for %i2 = 0 to 10 {
    affine.for %i3 = 0 to 100 {
      %v3 = affine.load %b[%i2] : memref<10xf32>
      %v4 = affine.load %a[%i2, %i3] : memref<10x100xf32>
      %v5 = arith.subf %v4, %v3 : f32
      affine.store %v5, %b[%i2] : memref<10xf32>
    }
  }
  // This test should fuse the src reduction loop at depth 1 in the destination
  // loop nest, which improves locality and enables subsequence passes to
  // decrease the reduction memref size and possibly place it in a faster
  // memory space.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 100 {
  // CHECK-NEXT:      affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x100xf32>
  // CHECK-NEXT:      "maxf"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 100 {
  // CHECK-NEXT:      affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x100xf32>
  // CHECK-NEXT:      arith.subf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_at_src_depth1_and_dst_depth1
func.func @should_fuse_at_src_depth1_and_dst_depth1() {
  %a = memref.alloc() : memref<100x16xf32>
  %b = memref.alloc() : memref<100x16xf32>

  affine.for %i0 = 0 to 100 {
    affine.for %i1 = 0 to 16 {
      %v0 = affine.load %a[%i0, %i1] : memref<100x16xf32>
      "op0"(%v0) : (f32) -> ()
    }
    affine.for %i2 = 0 to 16 {
      %v1 = "op1"() : () -> (f32)
      affine.store %v1, %b[%i0, %i2] : memref<100x16xf32>
    }
  }

  affine.for %i3 = 0 to 100 {
    affine.for %i4 = 0 to 16 {
      %v2 = affine.load %b[%i3, %i4] : memref<100x16xf32>
      "op2"(%v2) : (f32) -> ()
    }
  }
  // We can slice iterations of the '%i0' and '%i1' loops in the source
  // loop nest, but slicing at depth 2 and inserting the slice in the
  // destination loop nest at depth2 causes extra computation. Instead,
  // the fusion algorithm should detect that the source loop should be sliced
  // at depth 1 and the slice should be inserted at depth 1.
  // CHECK:       affine.for %{{.*}} = 0 to 100 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 16 {
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<100x16xf32>
  // CHECK-NEXT:      "op0"(%{{.*}}) : (f32) -> ()
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 16 {
  // CHECK-NEXT:      %{{.*}} = "op1"() : () -> f32
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0, %{{.*}}] : memref<1x16xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 16 {
  // CHECK-NEXT:      affine.load %{{.*}}[0, %{{.*}}] : memref<1x16xf32>
  // CHECK-NEXT:      "op2"(%{{.*}}) : (f32) -> ()
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----
// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<(d0, d1) -> (d0 * 10 + d1)>

// CHECK-LABEL: func @should_fuse_src_depth1_at_dst_depth2
func.func @should_fuse_src_depth1_at_dst_depth2() {
  %a = memref.alloc() : memref<100xf32>
  %c0 = arith.constant 0.0 : f32

  affine.for %i0 = 0 to 100 {
    affine.store %c0, %a[%i0] : memref<100xf32>
  }

  affine.for %i1 = 0 to 10 {
    affine.for %i2 = 0 to 10 {
      %a0 = affine.apply affine_map<(d0, d1) -> (d0 * 10 + d1)> (%i1, %i2)
      %v0 = affine.load %a[%a0] : memref<100xf32>
    }
  }
  // The source loop nest slice loop bound is a function of both destination
  // loop IVs, so we should slice at depth 1 and insert the slice at depth 2.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:      affine.apply [[$MAP0]](%{{.*}}, %{{.*}})
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:      affine.apply [[$MAP0]](%{{.*}}, %{{.*}})
  // CHECK-NEXT:      affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @fusion_at_depth0_not_currently_supported
func.func @fusion_at_depth0_not_currently_supported() {
  %0 = memref.alloc() : memref<10xf32>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %0[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %0[%c0] : memref<10xf32>
  }
  // NOTE: Should shrink memref size to 1 element access by load in dst loop
  // nest, and make the store in the slice store to the same element.
  // CHECK-DAG:   memref.alloc() : memref<1xf32>
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_deep_loop_nests
func.func @should_fuse_deep_loop_nests() {
  %0 = memref.alloc() : memref<2x2x3x3x16x10xf32, 2>
  %1 = memref.alloc() : memref<2x2x3x3x16x10xf32, 2>
  %2 = memref.alloc() : memref<3x3x3x3x16x10xf32, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_0 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 2 {
    affine.for %i1 = 0 to 2 {
      affine.for %i2 = 0 to 3 {
        affine.for %i3 = 0 to 3 {
          affine.for %i4 = 0 to 16 {
            affine.for %i5 = 0 to 10 {
              %3 = affine.load %0[%i0, %i1, %i2, %i3, %i4, %i5]
                : memref<2x2x3x3x16x10xf32, 2>
            }
          }
          affine.for %i6 = 0 to 16 {
            affine.for %i7 = 0 to 10 {
              affine.store %cst, %1[%i0, %i1, %i2, %i3, %i6, %i7]
                : memref<2x2x3x3x16x10xf32, 2>
            }
          }
        }
      }
    }
  }
  affine.for %i8 = 0 to 3 {
    affine.for %i9 = 0 to 3 {
      affine.for %i10 = 0 to 2 {
        affine.for %i11 = 0 to 2 {
          affine.for %i12 = 0 to 3 {
            affine.for %i13 = 0 to 3 {
              affine.for %i14 = 0 to 2 {
                affine.for %i15 = 0 to 2 {
                  affine.for %i16 = 0 to 16 {
                    affine.for %i17 = 0 to 10 {
                      %5 = affine.load %0[%i14, %i15, %i12, %i13, %i16, %i17]
                        : memref<2x2x3x3x16x10xf32, 2>
                    }
                  }
                  affine.for %i18 = 0 to 16 {
                    affine.for %i19 = 0 to 10 {
                      %6 = affine.load %1[%i10, %i11, %i8, %i9, %i18, %i19]
                        : memref<2x2x3x3x16x10xf32, 2>
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
// The first four loops of the source loop nest can be sliced with iteration
// bounds which are a function of the first four loops of destination loop nest,
// where the destination loops nests have been interchanged.

// CHECK-DAG:   memref.alloc() : memref<1x1x1x1x16x10xf32, 2>
// CHECK:       affine.for %{{.*}} = 0 to 3 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 3 {
// CHECK-NEXT:      affine.for %{{.*}} = 0 to 2 {
// CHECK-NEXT:        affine.for %{{.*}} = 0 to 2 {
// CHECK-NEXT:          affine.for %{{.*}} = 0 to 3 {
// CHECK-NEXT:            affine.for %{{.*}} = 0 to 3 {
// CHECK-NEXT:              affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:                affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:                  affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<2x2x3x3x16x10xf32, 2>
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:              affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:                affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:                  affine.store %{{.*}}, %{{.*}}[0, 0, 0, 0, %{{.*}}, %{{.*}}] : memref<1x1x1x1x16x10xf32, 2>
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:              affine.for %{{.*}} = 0 to 2 {
// CHECK-NEXT:                affine.for %{{.*}} = 0 to 2 {
// CHECK-NEXT:                  affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:                    affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:                      affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<2x2x3x3x16x10xf32, 2>
// CHECK-NEXT:                    }
// CHECK-NEXT:                  }
// CHECK-NEXT:                  affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:                    affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:                      affine.load %{{.*}}[0, 0, 0, 0, %{{.*}}, %{{.*}}] : memref<1x1x1x1x16x10xf32, 2>
// CHECK-NEXT:                    }
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_at_depth1_and_reduce_slice_trip_count
func.func @should_fuse_at_depth1_and_reduce_slice_trip_count() {
  %a = memref.alloc() : memref<4x256xf32>
  %b = memref.alloc() : memref<4x256xf32>

  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32

  affine.for %i0 = 0 to 4 {
    affine.for %i1 = 0 to 256 {
      %v0 = affine.load %b[%i0, %i1] : memref<4x256xf32>
    }
    affine.for %i2 = 0 to 256 {
      affine.store %cf0, %a[%i0, %i2] : memref<4x256xf32>
    }
  }

  affine.for %d0 = 0 to 4 {
    affine.for %d1 = 0 to 16 {
      %v1 = affine.load %a[%d0, %d1] : memref<4x256xf32>
    }
  }
  // The cost of fusing at depth 2 is greater than the cost of fusing at depth 1
  // for two reasons:
  // 1) Inserting the unsliceable src loop %i1 to a higher depth removes
  //    redundant computation and reduces costs.
  // 2) Inserting the sliceable src loop %i2 at depth 1, we can still reduce
  //    its trip count to 16 (from 256) reducing costs.
  // NOTE: the size of the private memref created for the fused loop nest
  // is reduced from the original shape from 4x256 to 4x16 because of the
  // data accessed by the load.
  // CHECK-DAG:   memref.alloc() : memref<1x16xf32>
  // CHECK:       affine.for %{{.*}} = 0 to 4 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 256 {
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x256xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 16 {
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0, %{{.*}}] : memref<1x16xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 16 {
  // CHECK-NEXT:      affine.load %{{.*}}[0, %{{.*}}] : memref<1x16xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_at_depth1_with_trip_count_20
func.func @should_fuse_at_depth1_with_trip_count_20() {
  %a = memref.alloc() : memref<100xf32>
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32

  affine.for %i0 = 0 to 100 {
    affine.store %cf0, %a[%i0]: memref<100xf32>
  }

  affine.for %i1 = 0 to 5 {
    affine.for %i2 = 0 to 10 {
      %v0 = affine.load %a[%i2]: memref<100xf32>
    }
    affine.for %i3 = 0 to 10 {
      affine.for %i4 = 0 to 20 {
        %v1 = affine.load %a[%i4]: memref<100xf32>
      }
    }
  }
  // NOTE: The size of the private memref created for fusion is shrunk to 20xf32
  // CHECK-DAG:   memref.alloc() : memref<20xf32>
  // CHECK:       affine.for %{{.*}} = 0 to 5 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 20 {
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<20xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}] : memref<20xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:      affine.for %{{.*}} = 0 to 20 {
  // CHECK-NEXT:        affine.load %{{.*}}[%{{.*}}] : memref<20xf32>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_at_depth1_with_trip_count_19
func.func @should_fuse_at_depth1_with_trip_count_19() {
  %a = memref.alloc() : memref<100xf32>
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32

  affine.for %i0 = 0 to 100 {
    affine.store %cf0, %a[%i0]: memref<100xf32>
  }

  affine.for %i1 = 0 to 5 {
    affine.for %i2 = 0 to 19 {
      %v0 = affine.load %a[%i2]: memref<100xf32>
    }
    affine.for %i3 = 0 to 10 {
      affine.for %i4 = 0 to 10 {
        %v1 = affine.load %a[%i4]: memref<100xf32>
      }
    }
  }
  // NOTE: The size of the private memref created for fusion is shrunk to 19xf32
  // CHECK-DAG:   memref.alloc() : memref<19xf32>
  // CHECK:       affine.for %{{.*}} = 0 to 5 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 19 {
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<19xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 19 {
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}] : memref<19xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:        affine.load %{{.*}}[%{{.*}}] : memref<19xf32>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}


// -----

// CHECK-LABEL: func @should_fuse_with_private_memref() {
func.func @should_fuse_with_private_memref() {
  %m = memref.alloc() : memref<100xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 100 {
    affine.store %cf7, %m[%i0] : memref<100xf32>
  }
  affine.for %i1 = 0 to 17 {
    %v0 = affine.load %m[%i1] : memref<100xf32>
  }
  affine.for %i2 = 0 to 82 {
    %v1 = affine.load %m[%i2] : memref<100xf32>
  }
  // Should create a new private memref.
  // CHECK-DAG:  memref.alloc() : memref<1xf32>
  // CHECK:      affine.for %{{.*}} = 0 to 17 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_live_out_arg_but_preserve_src_loop(%{{.*}}: memref<10xf32>) {
func.func @should_fuse_live_out_arg_but_preserve_src_loop(%arg0: memref<10xf32>) {
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %arg0[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 9 {
    %v0 = affine.load %arg0[%i1] : memref<10xf32>
  }
  // This tests that the loop nest '%i0' should not be removed after fusion
  // because it writes to memref argument '%arg0', and its read region
  // does not cover its write region (so fusion would shrink the write region
  // in the fused loop nest, so complete live out data region would not
  // be written).
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}} : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 9 {
  // CHECK-NEXT:    affine.store %{{.*}} : memref<1xf32>
  // CHECK-NEXT:    affine.load %{{.*}} : memref<1xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_live_out_arg(%{{.*}}: memref<10xf32>) {
func.func @should_fuse_live_out_arg(%arg0: memref<10xf32>) {
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %arg0[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %arg0[%i1] : memref<10xf32>
  }
  // The read/write regions for memref '%{{.*}}' are the same for both
  // loops, so they should fuse.

  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_escaping_memref_but_preserve_src_loop() -> memref<10xf32>
func.func @should_fuse_escaping_memref_but_preserve_src_loop() -> memref<10xf32> {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 9 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }
  // This tests that the loop nest '%i0' should not be removed after fusion
  // because it writes to memref '%m', which is returned by the function, and
  // the '%i1' memory region does not cover '%i0' memory region.

  // CHECK-DAG:   memref.alloc() : memref<1xf32>
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}} : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 9 {
  // CHECK-NEXT:    affine.store %{{.*}} : memref<1xf32>
  // CHECK-NEXT:    affine.load %{{.*}} : memref<1xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return %{{.*}} : memref<10xf32>
  return %m : memref<10xf32>
}
// -----

// This should fuse with the %in becoming a 1x1x1.
func.func @R3_to_R2_reshape() {
  %in = memref.alloc() : memref<2x3x16xi32>

  %c0 = arith.constant 0 : index

  affine.for %i0 = 0 to 2 {
    affine.for %i1 = 0 to 3 {
      affine.for %i2 = 0 to 16 {
        %val = "foo"(%i0, %i1, %i2) : (index, index, index) -> i32
        affine.store %val, %in[%i0, %i1, %i2] : memref<2x3x16xi32>
      }
    }
  }

  affine.for %ii = 0 to 32 {
    affine.for %jj = 0 to 3 {
      %a0 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)> (%ii, %jj)
      %idx = affine.apply affine_map<(d0) -> (d0 floordiv (3 * 16))> (%a0)
      %v = affine.load %in[%idx, %jj, %c0]
        : memref<2x3x16xi32>
    }
  }
  return
}
// CHECK-DAG: [[$MAP0:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> ((d0 * 3 + d1) floordiv 48)>
// CHECK-DAG: [[$MAP1:#map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (d0 * 3 + d1)>
// CHECK-DAG: [[$MAP2:#map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 floordiv 48)>

// CHECK-LABEL: func @R3_to_R2_reshape()
// CHECK-DAG:    memref.alloc() : memref<1x1x1xi32>
// CHECK:        affine.for %{{.*}} = 0 to 32 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 3 {
// CHECK-NEXT:      affine.apply [[$MAP0]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      "foo"(%{{.*}}, %{{.*}}, %{{.*}}) : (index, index, index) -> i32
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0, 0, 0] : memref<1x1x1xi32>
// CHECK-NEXT:      affine.apply [[$MAP1]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP2]](%{{.*}})
// CHECK-NEXT:      affine.load %{{.*}}[0, 0, 0] : memref<1x1x1xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// -----

func.func @should_fuse_multi_output_producer() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>

  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %a[%i0] : memref<10xf32>
    affine.store %cf7, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %a[%i1] : memref<10xf32>
    %v1 = affine.load %b[%i1] : memref<10xf32>
  }

  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @fusion_preventing_deps_on_middle_loop() {
func.func @fusion_preventing_deps_on_middle_loop() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %a[%i0] : memref<10xf32>
    affine.store %v0, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %a[%i1] : memref<10xf32>
    %v1 = affine.load %c[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v2 = affine.load %b[%i2] : memref<10xf32>
    affine.store %v2, %c[%i2] : memref<10xf32>
  }
  // Loops '%i0' and '%i2' cannot fuse along producer/consumer edge on memref
  // '%b', because of the WAR dep from '%i0' to '%i1' on memref '%a' and
  // because of the WAR dep from '%i1' to '%i2' on memref '%c'.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_and_move_to_preserve_war_dep() {
func.func @should_fuse_and_move_to_preserve_war_dep() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %b[%i0] : memref<10xf32>
    affine.store %v0, %a[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 3 {
    %v2 = affine.load %c[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 5 {
    affine.store %cf7, %b[%i2] : memref<10xf32>
  }
  affine.for %i3 = 0 to 10 {
    %v1 = affine.load %a[%i3] : memref<10xf32>
    affine.store %cf7, %c[%i3] : memref<10xf32>
  }

  // Dependence graph:
  //
  //         %i0 ---------
  //               |     |
  //     --- %i1   | %b  | %a
  //     |         |     |
  //  %c |   %i2 <--     |
  //     |               |
  //     --> %i3 <--------
  //
  // It is possible to fuse loop '%i0' into '%i3' and preserve dependences
  // if the fused loop nest is inserted between loops '%i1' and '%i2'.

  // CHECK-DAG:   memref.alloc() : memref<1xf32>
  // CHECK:       affine.for %{{.*}} = 0 to 3 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 5 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @fusion_preventing_dep_on_constant() {
func.func @fusion_preventing_dep_on_constant() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %b[%i0] : memref<10xf32>
    affine.store %cf7, %a[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %b[%i1] : memref<10xf32>
  }
  %cf11 = arith.constant 11.0 : f32
  affine.for %i2 = 0 to 10 {
    %v2 = affine.load %a[%i2] : memref<10xf32>
    affine.store %cf11, %c[%i2] : memref<10xf32>
  }
  // Loops '%i0' and '%i2' cannot fuse along producer/consumer edge on memref
  // '%a', because of the WAR dep from '%i0' to '%i1' on memref '%b' and
  // because of the SSA value dep from '%cf11' def to use in '%i2'.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %{{.*}} = arith.constant 1.100000e+01 : f32
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_and_preserve_dep_on_constant() {
func.func @should_fuse_and_preserve_dep_on_constant() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = arith.constant 7.0 : f32
  %cf11 = arith.constant 11.0 : f32
  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %b[%i0] : memref<10xf32>
    affine.store %cf7, %a[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %b[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v2 = affine.load %a[%i2] : memref<10xf32>
    affine.store %cf11, %c[%i2] : memref<10xf32>
  }

  // Loops '%i0' and '%i2' can fuse along producer/consumer edge on memref
  // '%a', and preserve the WAR dep from '%i0' to '%i1' on memref '%b', and
  // the SSA value dep from '%cf11' def to use in '%i2'.

  // CHECK:       arith.constant 1.100000e+01 : f32
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: @producer_consumer_with_outmost_user
func.func @producer_consumer_with_outmost_user(%arg0 : f16) {
  %c0 = arith.constant 0 : index
  %src = memref.alloc() : memref<f16, 1>
  %dst = memref.alloc() : memref<f16>
  %tag = memref.alloc() : memref<1xi32>
  affine.for %arg1 = 4 to 6 {
    affine.for %arg2 = 0 to 1 {
      %0 = arith.addf %arg0, %arg0 : f16
      affine.store %0, %src[] : memref<f16, 1>
    }
    affine.for %arg3 = 0 to 1 {
      %0 = affine.load %src[] : memref<f16, 1>
    }
  }
  affine.dma_start %src[], %dst[], %tag[%c0], %c0 : memref<f16, 1>, memref<f16>, memref<1xi32>
  // CHECK:       %[[CST_INDEX:.*]] = arith.constant 0 : index
  // CHECK:       %[[DMA_SRC:.*]] = memref.alloc() : memref<f16, 1>
  // CHECK:       %[[DMA_DST:.*]] = memref.alloc() : memref<f16>
  // CHECK:       %[[DMA_TAG:.*]] = memref.alloc() : memref<1xi32>
  // CHECK:       affine.for %arg1 = 4 to 6
  // CHECK-NEXT:  affine.for %arg2 = 0 to 1
  // CHECK-NEXT:  %[[RESULT_ADD:.*]] = arith.addf %arg0, %arg0 : f16
  // CHECK-NEXT:  affine.store %[[RESULT_ADD]], %[[DMA_SRC]][] : memref<f16, 1>
  // CHECK-NEXT:  affine.load %[[DMA_SRC]][] : memref<f16, 1>
  // CHECK:       affine.dma_start %[[DMA_SRC]][], %[[DMA_DST]][], %[[DMA_TAG]][%[[CST_INDEX]]], %[[CST_INDEX]] : memref<f16, 1>, memref<f16>, memref<1xi32>
  // CHECK-NEXT:  return
  return
}

// Add further tests in mlir/test/Transforms/loop-fusion-4.mlir


// RUN: mlir-opt -allow-unregistered-dialect %s -affine-loop-fusion -split-input-file | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -affine-loop-fusion="fusion-maximal" -split-input-file | FileCheck %s --check-prefix=MAXIMAL

// TODO: Add more tests:
// *) Add nested fusion test cases when non-constant loop bound support is
//    added to iteration domain in dependence check.
// *) Add a test w/ floordiv/ceildiv/mod when supported in dependence check.
// *) Add tests which check fused computation slice indexing and loop bounds.
// TODO: Test clean up: move memref allocs to func args.

// -----

// CHECK-LABEL: func @should_fuse_raw_dep_for_locality() {
func @should_fuse_raw_dep_for_locality() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

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
func @should_fuse_reduction_to_pointwise() {
  %a = memref.alloc() : memref<10x10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %v0 = affine.load %b[%i0] : memref<10xf32>
      %v1 = affine.load %a[%i0, %i1] : memref<10x10xf32>
      %v3 = addf %v0, %v1 : f32
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
  // CHECK-NEXT:      addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-DAG: [[$MAP_SHIFT_MINUS_ONE_R1:#map[0-9]+]] = affine_map<(d0) -> (d0 - 1)>
// CHECK-DAG: [[$MAP_SHIFT_D0_BY_ONE:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + 1)>
// CHECK-DAG: [[$MAP_SHIFT_D1_BY_ONE:#map[0-9]+]] = affine_map<(d0, d1) -> (d1 + 1)>

// CHECK-LABEL: func @should_fuse_loop_nests_with_shifts() {
func @should_fuse_loop_nests_with_shifts() {
  %a = memref.alloc() : memref<10x10xf32>
  %cf7 = constant 7.0 : f32

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
func @should_fuse_loop_nest() {
  %a = memref.alloc() : memref<10x10xf32>
  %b = memref.alloc() : memref<10x10xf32>
  %cf7 = constant 7.0 : f32

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
  // CHECK-DAG:  [[NEWA:%[0-9]+]] = memref.alloc() : memref<1x1xf32>
  // CHECK-DAG:  [[NEWB:%[0-9]+]] = memref.alloc() : memref<1x1xf32>
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
func @should_fuse_across_intermediate_loop_with_no_deps() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

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
func @should_fuse_all_loops() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

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
  // CHECK-DAG: [[NEWA:%[0-9]+]] = memref.alloc() : memref<1xf32>
  // CHECK-DAG: [[NEWB:%[0-9]+]] = memref.alloc() : memref<1xf32>
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
func @should_fuse_first_and_second_loops() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

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
func @should_not_fuse_would_create_cycle() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

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
func @should_fuse_producer_consumer() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

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
func @should_fuse_and_move_to_preserve_war_dep() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

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
func @should_fuse_if_top_level_access() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }

  %c0 = constant 4 : index
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
func @should_fuse_but_not_remove_src() {
  %m = memref.alloc() : memref<100xf32>
  %cf7 = constant 7.0 : f32

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
func @should_fuse_no_top_level_access() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

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

// CHECK-LABEL: func @should_not_fuse_if_inst_at_top_level() {
func @should_not_fuse_if_inst_at_top_level() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }
  %c0 = constant 4 : index
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

// CHECK-LABEL: func @should_not_fuse_if_inst_in_loop_nest() {
func @should_not_fuse_if_inst_in_loop_nest() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %c4 = constant 4 : index

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.if #set0(%c4) {
    }
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }

  // IfOp in ForInst should prevent fusion.
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

// CHECK-LABEL: func @permute_and_fuse() {
func @permute_and_fuse() {
  %m = memref.alloc() : memref<10x20x30xf32>

  %cf7 = constant 7.0 : f32
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

// CHECK-DAG: [[$MAP0:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>
// CHECK-DAG: [[$MAP1:#map[0-9]+]] = affine_map<(d0) -> (d0 floordiv 4)>
// CHECK-DAG: [[$MAP2:#map[0-9]+]] = affine_map<(d0) -> (d0 mod 4)>

// Reshape from a 64 x f32 to 16 x 4 x f32.
// CHECK-LABEL: func @fuse_reshape_64_16_4
func @fuse_reshape_64_16_4(%in : memref<64xf32>) {
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
// CHECK-DAG: [[$MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0 floordiv 4)>
// CHECK-DAG: [[$MAP1:#map[0-9]+]] = affine_map<(d0) -> (d0 mod 4)>
// CHECK-DAG: [[$MAP2:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>

// Reshape a 16x4xf32 to 64xf32.
// CHECK-LABEL: func @fuse_reshape_16_4_64
func @fuse_reshape_16_4_64() {
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
func @R6_to_R2_reshape_square() -> memref<64x9xi32> {
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
      %b = muli %a, %a : i32
      affine.store %b, %live_out[%i, %j] : memref<64x9xi32>
    }
  }
  return %live_out : memref<64x9xi32>
}
// Everything above is fused to a single 2-d loop nest, and the 6-d tensor %in
// is eliminated if -memref-dataflow-opt is also supplied.
//
// CHECK-DAG: [[$MAP0:#map[0-9]+]] = affine_map<(d0, d1) -> ((d0 * 9 + d1) floordiv 288)>
// CHECK-DAG: [[$MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (((d0 * 9 + d1) mod 288) floordiv 144)>
// CHECK-DAG: [[$MAP2:#map[0-9]+]] = affine_map<(d0, d1) -> ((((d0 * 9 + d1) mod 288) mod 144) floordiv 48)>
// CHECK-DAG: [[$MAP3:#map[0-9]+]] = affine_map<(d0, d1) -> (((((d0 * 9 + d1) mod 288) mod 144) mod 48) floordiv 16)>
// CHECK-DAG: [[$MAP4:#map[0-9]+]] = affine_map<(d0, d1) -> (((((d0 * 9 + d1) mod 288) mod 144) mod 48) mod 16)>
// CHECK-DAG: [[$MAP11:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 * 9 + d1)>
// CHECK-DAG: [[$MAP12:#map[0-9]+]] = affine_map<(d0) -> (d0 floordiv 288)>
// CHECK-DAG: [[$MAP13:#map[0-9]+]] = affine_map<(d0) -> ((d0 mod 288) floordiv 144)>
// CHECK-DAG: [[$MAP14:#map[0-9]+]] = affine_map<(d0) -> (((d0 mod 288) mod 144) floordiv 48)>
// CHECK-DAG: [[$MAP15:#map[0-9]+]] = affine_map<(d0) -> ((((d0 mod 288) mod 144) mod 48) floordiv 16)>
// CHECK-DAG: [[$MAP16:#map[0-9]+]] = affine_map<(d0) -> ((((d0 mod 288) mod 144) mod 48) mod 16)>
// CHECK-DAG: [[$MAP17:#map[0-9]+]] = affine_map<(d0) -> (0)>

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
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0, ((%{{.*}} * 9 + %{{.*}}) mod 288) floordiv 144, (((%{{.*}} * 9 + %{{.*}}) mod 288) mod 144) floordiv 48, ((((%{{.*}} * 9 + %{{.*}}) mod 288) mod 144) mod 48) floordiv 16, ((((%{{.*}} * 9 + %{{.*}}) mod 288) mod 144) mod 48) mod 16, 0] : memref<1x2x3x3x16x1xi32>
// CHECK-NEXT:      affine.apply [[$MAP11]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP12]](%{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP13]](%{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP14]](%{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP15]](%{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP16]](%{{.*}})
// CHECK-NEXT:      affine.apply [[$MAP17]](%{{.*}})
// CHECK-NEXT:      affine.load %{{.*}}[0, ((%{{.*}} * 9 + %{{.*}}) mod 288) floordiv 144, (((%{{.*}} * 9 + %{{.*}}) mod 288) mod 144) floordiv 48, ((((%{{.*}} * 9 + %{{.*}}) mod 288) mod 144) mod 48) floordiv 16, ((((%{{.*}} * 9 + %{{.*}}) mod 288) mod 144) mod 48) mod 16, 0] : memref<1x2x3x3x16x1xi32>
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xi32>
// CHECK-NEXT:      affine.load %{{.*}}[0, 0] : memref<1x1xi32>
// CHECK-NEXT:      muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<64x9xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return %{{.*}} : memref<64x9xi32>

// -----

// CHECK-LABEL: func @fuse_symbolic_bounds
func @fuse_symbolic_bounds(%M : index, %N : index) {
  %N_plus_5 = affine.apply affine_map<(d0) -> (d0 + 5)>(%N)
  %m = memref.alloc(%M, %N_plus_5) : memref<? x ? x f32>

  %c0 = constant 0.0 : f32
  %s = constant 5 : index

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
func @should_fuse_reduction_at_depth_of_one() {
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
      %v5 = subf %v4, %v3 : f32
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
  // CHECK-NEXT:      subf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_at_src_depth1_and_dst_depth1
func @should_fuse_at_src_depth1_and_dst_depth1() {
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
func @should_fuse_src_depth1_at_dst_depth2() {
  %a = memref.alloc() : memref<100xf32>
  %c0 = constant 0.0 : f32

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
func @fusion_at_depth0_not_currently_supported() {
  %0 = memref.alloc() : memref<10xf32>
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
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
func @should_fuse_deep_loop_nests() {
  %0 = memref.alloc() : memref<2x2x3x3x16x10xf32, 2>
  %1 = memref.alloc() : memref<2x2x3x3x16x10xf32, 2>
  %2 = memref.alloc() : memref<3x3x3x3x16x10xf32, 2>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c1_0 = constant 1 : index
  %cst = constant 0.000000e+00 : f32
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
func @should_fuse_at_depth1_and_reduce_slice_trip_count() {
  %a = memref.alloc() : memref<4x256xf32>
  %b = memref.alloc() : memref<4x256xf32>

  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32

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
func @should_fuse_at_depth1_with_trip_count_20() {
  %a = memref.alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32

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
func @should_fuse_at_depth1_with_trip_count_19() {
  %a = memref.alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32

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

// CHECK-LABEL: func @should_fuse_with_private_memrefs_with_diff_shapes() {
func @should_fuse_with_private_memrefs_with_diff_shapes() {
  %m = memref.alloc() : memref<100xf32>
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 100 {
    affine.store %cf7, %m[%i0] : memref<100xf32>
  }
  affine.for %i1 = 0 to 17 {
    %v0 = affine.load %m[%i1] : memref<100xf32>
  }
  affine.for %i2 = 0 to 82 {
    %v1 = affine.load %m[%i2] : memref<100xf32>
  }
  // Should create two new private memrefs customized to the shapes accessed
  // by loops %{{.*}} and %{{.*}}.
  // CHECK-DAG:  memref.alloc() : memref<1xf32>
  // CHECK-DAG:  memref.alloc() : memref<1xf32>
  // CHECK:      affine.for %{{.*}} = 0 to 17 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 82 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_live_out_arg_but_preserve_src_loop(%{{.*}}: memref<10xf32>) {
func @should_fuse_live_out_arg_but_preserve_src_loop(%arg0: memref<10xf32>) {
  %cf7 = constant 7.0 : f32

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
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 9 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_live_out_arg(%{{.*}}: memref<10xf32>) {
func @should_fuse_live_out_arg(%arg0: memref<10xf32>) {
  %cf7 = constant 7.0 : f32

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
func @should_fuse_escaping_memref_but_preserve_src_loop() -> memref<10xf32> {
  %cf7 = constant 7.0 : f32
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

  // CHECK-DAG:   memref.alloc() : memref<10xf32>
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 9 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return %{{.*}} : memref<10xf32>
  return %m : memref<10xf32>
}
// -----

// This should fuse with the %in becoming a 1x1x1.
func @R3_to_R2_reshape() {
  %in = memref.alloc() : memref<2x3x16xi32>

  %c0 = constant 0 : index

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
// CHECK-DAG: [[$MAP0:#map[0-9]+]] = affine_map<(d0, d1) -> ((d0 * 3 + d1) floordiv 48)>
// CHECK-DAG: [[$MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 * 3 + d1)>
// CHECK-DAG: [[$MAP2:#map[0-9]+]] = affine_map<(d0) -> (d0 floordiv 48)>

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

func @should_fuse_multi_output_producer() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

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
func @fusion_preventing_deps_on_middle_loop() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

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
func @should_fuse_and_move_to_preserve_war_dep() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

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
func @fusion_preventing_dep_on_constant() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %b[%i0] : memref<10xf32>
    affine.store %cf7, %a[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %b[%i1] : memref<10xf32>
  }
  %cf11 = constant 11.0 : f32
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
  // CHECK-NEXT:  %{{.*}} = constant 1.100000e+01 : f32
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_and_preserve_dep_on_constant() {
func @should_fuse_and_preserve_dep_on_constant() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32
  %cf11 = constant 11.0 : f32
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

  // CHECK:       constant 1.100000e+01 : f32
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

// CHECK-LABEL: func @should_fuse_at_depth_above_loop_carried_dependence(%{{.*}}: memref<64x4xf32>, %{{.*}}: memref<64x4xf32>) {
func @should_fuse_at_depth_above_loop_carried_dependence(%arg0: memref<64x4xf32>, %arg1: memref<64x4xf32>) {
  %out = memref.alloc() : memref<64x4xf32>
  %0 = constant 0.0 : f32
  affine.for %i0 = 0 to 64 {
    affine.for %i1 = 0 to 4 {
      affine.store %0, %out[%i0, %i1] : memref<64x4xf32>
    }
  }
  affine.for %i2 = 0 to 4 {
    affine.for %i3 = 0 to 4 {
      affine.for %i4 = 0 to 16 {
        %v = affine.load %arg1[16 * %i3 - %i4 + 15, %i2] : memref<64x4xf32>
        "op0"(%v) : (f32) -> ()
      }
      affine.for %i5 = 0 to 4 {
        affine.for %i6 = 0 to 16 {
          %v = affine.load %arg0[16 * %i5 - %i6 + 15, %i3] : memref<64x4xf32>
          "op1"(%v) : (f32) -> ()
        }
        affine.for %i7 = 0 to 16 {
          %r = "op2"() : () -> (f32)
          %v = affine.load %out[16 * %i5 + %i7, %i2] : memref<64x4xf32>
          %s = addf %v, %r : f32
          affine.store %s, %out[16 * %i5 + %i7, %i2] : memref<64x4xf32>
        }
      }
    }
  }

  // We can fuse source loop nest '%i0' into dst loop nest '%i2', but the
  // depth at which we can insert the src loop nest slice into the dst loop
  // lest must be decreased because of a loop carried dependence on loop '%i3'.
  // As a result, the source loop nest is inserted at dst loop nest depth 1,
  // just above the loop with the carried dependence. In addition, the source
  // loop nest iteration bounds on its loop '%i1' are reduced to 1, so the
  // memref size can be reduced to 128x1xf32.

  // CHECK:       memref.alloc() : memref<64x1xf32>
  // CHECK:       affine.for %{{.*}} = 0 to 4 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 64 {
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}, 0] : memref<64x1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 4 {
  // CHECK-NEXT:      affine.for %{{.*}} = 0 to 16 {
  // CHECK-NEXT:        affine.load %{{.*}}[%{{.*}} * 16 - %{{.*}} + 15, %{{.*}}] : memref<64x4xf32>
  // CHECK-NEXT:        "op0"(%{{.*}}) : (f32) -> ()
  // CHECK-NEXT:      }
  // CHECK-NEXT:      affine.for %{{.*}} = 0 to 4 {
  // CHECK-NEXT:        affine.for %{{.*}} = 0 to 16 {
  // CHECK-NEXT:          affine.load %{{.*}}[%{{.*}} * 16 - %{{.*}} + 15, %{{.*}}] : memref<64x4xf32>
  // CHECK-NEXT:          "op1"(%{{.*}}) : (f32) -> ()
  // CHECK-NEXT:        }
  // CHECK-NEXT:        affine.for %{{.*}} = 0 to 16 {
  // CHECK-NEXT:          %{{.*}} = "op2"() : () -> f32
  // CHECK:               affine.load %{{.*}}[%{{.*}} * 16 + %{{.*}}, 0] : memref<64x1xf32>
  // CHECK-NEXT:          addf %{{.*}}, %{{.*}} : f32
  // CHECK:               affine.store %{{.*}}, %{{.*}}[%{{.*}} * 16 + %{{.*}}, 0] : memref<64x1xf32>
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_only_two_loops_and_remove_producer() {
func @should_fuse_only_two_loops_and_remove_producer() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %a[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %a[%i1] : memref<10xf32>
    affine.store %v0, %b[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v1 = affine.load %a[%i2] : memref<10xf32>
    affine.store %v1, %b[%i2] : memref<10xf32>
  }

  // On the first visit to '%i2', the fusion algorithm can not fuse loop nest
  // '%i0' into '%i2' because of the dependences '%i0' and '%i2' each have on
  // '%i1'. Then, '%i0' is fused into '%i1' and no private memref is created for
  // memref '%a' to be able to remove '%i0' and still preserve the depencence on
  // '%a' with '%i2'.
  // TODO: Alternatively, we could fuse '%i0' into '%i1' with a private memref,
  // the dependence between '%i0' and '%i1' on memref '%a' would no longer exist,
  // and '%i0' could be fused into '%i2' as well. Note that this approach would
  // duplicate the computation in loop nest '%i0' to loop nests '%i1' and '%i2',
  // which would limit its profitability.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:   return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_after_one_loop_interchange() {
func @should_fuse_after_one_loop_interchange() {
  %a = memref.alloc() : memref<10xf32>

  %cf0 = constant 0.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cf0, %a[%i0] : memref<10xf32>
  }

  affine.for %i1 = 0 to 5 {
    affine.for %i2 = 0 to 10 {
      %v0 = affine.load %a[%i2] : memref<10xf32>
      affine.store %v0, %a[%i2] : memref<10xf32>
    }
  }

  // The dependence between the load and affine.store is carried on loop '%i1', and
  // cannot be fused with loop '%i0' without violating this dependence.
  // Once loops '%i1' and %i2' are interchanged, loop '%i0' can be fused
  // at loop depth 1, because the loop carrying the dependence has been
  // interchanged and is now at depth 2.

  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 5 {
  // CHECK-NEXT:      affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_after_two_loop_interchanges() {
func @should_fuse_after_two_loop_interchanges() {
  %a = memref.alloc() : memref<6x8xf32>

  %cf0 = constant 0.0 : f32
  affine.for %i0 = 0 to 6 {
    affine.for %i1 = 0 to 8 {
      affine.store %cf0, %a[%i0, %i1] : memref<6x8xf32>
    }
  }

  affine.for %i2 = 0 to 4 {
    affine.for %i3 = 0 to 6 {
      affine.for %i4 = 0 to 2 {
        affine.for %i5 = 0 to 8 {
          %v0 = affine.load %a[%i3, %i5] : memref<6x8xf32>
          %v1 = addf %v0, %v0 : f32
          affine.store %v1, %a[%i3, %i5] : memref<6x8xf32>
        }
      }
    }
  }

  // The dependence between the load and affine.store is carried on loops '%i2' and
  // '%i4', and cannot be fused with loop '%i0' without violating this
  // dependence.
  // Once loop '%i2' is interchanged with loop '%i3', and again with loop
  // '%i5', then loop '%i0' can be fused at loop depth 2, because the loop
  // carrying the dependences have been interchanged with loops at depth > 2.

  // CHECK:       affine.for %{{.*}} = 0 to 6 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 8 {
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:      affine.for %{{.*}} = 0 to 4 {
  // CHECK-NEXT:        affine.for %{{.*}} = 0 to 2 {
  // CHECK-NEXT:          affine.load %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:          addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:          affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

func @should_fuse_live_out_writer(%arg0 : memref<10xf32>) -> memref<10xf32> {
  %cst = constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %arg0[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %arg0[%i1] : memref<10xf32>
    affine.store %1, %arg0[%i1] : memref<10xf32>
  }
  return %arg0 : memref<10xf32>

  // CHECK:       %{{.*}} = constant 0.000000e+00 : f32
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return %{{.*}} : memref<10xf32>
}

// -----

// The fused slice has 16 iterations from along %i0.

// CHECK-DAG: [[$MAP_LB:#map[0-9]+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG: [[$MAP_UB:#map[0-9]+]] = affine_map<(d0) -> (d0 * 16 + 16)>

// CHECK-LABEL: slice_tile
func @slice_tile(%arg0: memref<128x8xf32>, %arg1: memref<32x8xf32>, %0 : f32) -> memref<32x8xf32> {
  affine.for %i0 = 0 to 32 {
    affine.for %i1 = 0 to 8 {
      affine.store %0, %arg1[%i0, %i1] : memref<32x8xf32>
    }
  }
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 8 {
      affine.for %k = 0 to 8 {
        affine.for %kk = 0 to 16 {
          %v = affine.load %arg0[16 * %k + %kk, %j] : memref<128x8xf32>
          %r = "foo"(%v) : (f32) -> f32
        }
        affine.for %ii = 0 to 16 {
          %v = affine.load %arg1[16 * %i + %ii, %j] : memref<32x8xf32>
          %s = addf %v, %v : f32
          affine.store %s, %arg1[16 * %i + %ii, %j] : memref<32x8xf32>
        }
      }
    }
  }
  return %arg1 : memref<32x8xf32>
}
// CHECK:       affine.for %{{.*}} = 0 to 2 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 8 {
// CHECK-NEXT:      affine.for %{{.*}} = [[$MAP_LB]](%{{.*}}) to [[$MAP_UB]](%{{.*}}) {
// CHECK-NEXT:        affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<32x8xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %{{.*}} = 0 to 8 {
// CHECK-NEXT:        affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:          affine.load %{{.*}}[%{{.*}} * 16 + %{{.*}}, %{{.*}}] : memref<128x8xf32>
// CHECK-NEXT:          "foo"(%{{.*}}) : (f32) -> f32
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:          affine.load %{{.*}}[%{{.*}} * 16 + %{{.*}}, %{{.*}}] : memref<32x8xf32>
// CHECK-NEXT:          addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:          affine.store %{{.*}}, %{{.*}}[%{{.*}} * 16 + %{{.*}}, %{{.*}}] : memref<32x8xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return %{{.*}} : memref<32x8xf32>
// CHECK-NEXT:}

// -----

// Test case which illustrates fix for b/126454413
func @test_add_slice_bounds() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %c0 = constant 0 : index

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        %a0 = affine.apply affine_map<(d0) -> (d0)> (%i0)
        %a1 = affine.apply affine_map<(d0) -> (d0)> (%i0)
        %a2 = affine.apply affine_map<(d0, d1) -> (d0 - d1)> (%a0, %a1)
        affine.store %cf7, %a[%a2] : memref<10xf32>
      }
    }
  }
  affine.for %i3 = 0 to 10 {
    affine.for %i4 = 0 to 10 {
      affine.for %i5 = 0 to 10 {
        %v0 = affine.load %a[%c0] : memref<10xf32>
      }
    }
  }

// CHECK:        affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:       affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:         affine.apply #map0(%{{.*}})
// CHECK-NEXT:         affine.apply #map0(%{{.*}})
// CHECK-NEXT:         affine.apply #map1(%{{.*}}, %{{.*}})
// CHECK-NEXT:         affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:       affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
  return
}

// -----

func @should_fuse_init_loops_siblings_then_shared_producer(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) {
  %0 = memref.alloc() : memref<10x10xf32>
  %cst = constant 0.000000e+00 : f32
  %cst_0 = constant 1.000000e+00 : f32
  %cst_1 = constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.store %cst_1, %0[%i0, %i1] : memref<10x10xf32>
    }
  }
  affine.for %i2 = 0 to 3 {
    affine.for %i3 = 0 to 3 {
      affine.store %cst, %arg0[%i2, %i3] : memref<10x10xf32>
    }
  }
  affine.for %i4 = 0 to 3 {
    affine.for %i5 = 0 to 3 {
      %1 = affine.load %0[%i4, %i5] : memref<10x10xf32>
      %2 = affine.load %arg0[%i4, %i5] : memref<10x10xf32>
      %3 = mulf %1, %2 : f32
      affine.store %3, %arg0[%i4, %i5] : memref<10x10xf32>
    }
  }
  affine.for %i6 = 0 to 3 {
    affine.for %i7 = 0 to 3 {
      affine.store %cst_0, %arg1[%i6, %i7] : memref<10x10xf32>
    }
  }
  affine.for %i8 = 0 to 3 {
    affine.for %i9 = 0 to 3 {
      %4 = affine.load %0[%i8, %i9] : memref<10x10xf32>
      %5 = affine.load %arg1[%i8, %i9] : memref<10x10xf32>
      %6 = addf %4, %5 : f32
      affine.store %6, %arg1[%i8, %i9] : memref<10x10xf32>
    }
  }

  // Pass 1: should fuse single-use producer loop nests into their unique user,
  //         so '%i2' will fuse into '%i4' and '%i6' will fuse into '%i8'.
  // Pass 2: should fuse sibling loop nests which share no dependence edges,
  //         so should fuse '%i4' into '%i8'.
  // Pass 3: should fuse single-use producer loop nest '%i0' into '%i8'. Note
  //         that loop nest '%i0' now has a single user after Pass 2 fused its
  //         two users together).

// CHECK:        affine.for %{{.*}} = 0 to 3 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 3 {
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       affine.load %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       affine.load %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return

  return
}

// -----

func @two_matrix_vector_products() {
  %in_matrix = memref.alloc() : memref<10x10xf32>
  %in_vec0 = memref.alloc() : memref<10xf32>
  %in_vec1 = memref.alloc() : memref<10xf32>
  %out_vec0 = memref.alloc() : memref<10xf32>
  %out_vec1 = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  // Populate input matrix.
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.store %cf7, %in_matrix[%i0, %i1] : memref<10x10xf32>
    }
  }
  // out_vec0 = in_matrix x in_vec0
  affine.for %i2 = 0 to 10 {
    affine.for %i3 = 0 to 10 {
      %v0 = affine.load %in_matrix[%i2, %i3] : memref<10x10xf32>
      %v1 = affine.load %in_vec0[%i3] : memref<10xf32>
      %v2 = mulf %v0, %v1 : f32
      %v3 = affine.load %out_vec0[%i3] : memref<10xf32>
      %v4 = addf %v2, %v3 : f32
      affine.store %v4, %out_vec0[%i3] : memref<10xf32>
    }
  }
  // out_vec1 = in_matrix x in_vec1
  affine.for %i4 = 0 to 10 {
    affine.for %i5 = 0 to 10 {
      %v5 = affine.load %in_matrix[%i4, %i5] : memref<10x10xf32>
      %v6 = affine.load %in_vec1[%i5] : memref<10xf32>
      %v7 = mulf %v5, %v6 : f32
      %v8 = affine.load %out_vec1[%i5] : memref<10xf32>
      %v9 = addf %v7, %v8 : f32
      affine.store %v9, %out_vec1[%i5] : memref<10xf32>
    }
  }

// CHECK:        affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, 0] : memref<10x1xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}, 0] : memref<10x1xf32>
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:       mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:       addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}, 0] : memref<10x1xf32>
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:       mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:       addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
  return
}

// -----

func @should_not_slice_past_slice_barrier() {
  %0 = memref.alloc() : memref<100x16xf32>
  affine.for %i0 = 0 to 100 {
    affine.for %i1 = 0 to 16 {
      %1 = "op1"() : () -> f32
      affine.store %1, %0[%i0, %i1] : memref<100x16xf32>
    } {slice_fusion_barrier = true}
  }
  affine.for %i2 = 0 to 100 {
    affine.for %i3 = 0 to 16 {
      %2 = affine.load %0[%i2, %i3] : memref<100x16xf32>
      "op2"(%2) : (f32) -> ()
    }
  }
  // The 'slice_fusion_barrier' attribute on '%i1' prevents slicing the
  // iteration space of '%i1' and any enclosing loop nests.
// CHECK:        affine.for %{{.*}} = 0 to 100 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:       %{{.*}} = "op1"() : () -> f32
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[0, %{{.*}}] : memref<1x16xf32>
// CHECK-NEXT:     } {slice_fusion_barrier = true}
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:       affine.load %{{.*}}[0, %{{.*}}] : memref<1x16xf32>
// CHECK-NEXT:       "op2"(%{{.*}}) : (f32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
  return
}

// -----

#map0 = affine_map<(d0, d1) -> (d0 * 16 + d1)>
func @fuse_across_dim_mismatch(%arg0: memref<4x4x16x1xf32>, %arg1: memref<144x9xf32>, %arg2: memref<9xf32>) {
  %1 = memref.alloc() : memref<144x4xf32>
  %2 = constant 0.0 : f32
  affine.for %i2 = 0 to 9 {
    affine.for %i3 = 0 to 4 {
      affine.for %i5 = 0 to 16 {
        %7 = affine.apply #map0(%i2, %i5)
        affine.store %2, %1[%7, %i3] : memref<144x4xf32>
      }
    }
  }
  affine.for %i6 = 0 to 9 {
    affine.for %i7 = 0 to 9 {
      affine.for %i8 = 0 to 4 {
        affine.for %i10 = 0 to 16 {
          %10 = affine.apply #map0(%i6, %i10)
          %11 = affine.load %1[%10, %i8] : memref<144x4xf32>
        }
      }
    }
  }
  return
}
// MAXIMAL:      #map = affine_map<(d0, d1) -> (d0 * 16 + d1)>
// MAXIMAL-LABEL: func @fuse_across_dim_mismatch
// MAXIMAL:        memref.alloc() : memref<1x1xf32>
// MAXIMAL:        affine.for %{{.*}} = 0 to 9 {
// MAXIMAL-NEXT:    affine.for %{{.*}} = 0 to 9 {
// MAXIMAL-NEXT:      affine.for %{{.*}} = 0 to 4 {
// MAXIMAL-NEXT:        affine.for %{{.*}} = 0 to 16 {
// MAXIMAL-NEXT:          affine.apply #map(%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:          affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
// MAXIMAL-NEXT:          affine.apply #map(%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:          affine.load %{{.*}}[0, 0] : memref<1x1xf32>
// MAXIMAL-NEXT:        }
// MAXIMAL-NEXT:      }
// MAXIMAL-NEXT:    }
// MAXIMAL-NEXT:  }

// -----

#map3 = affine_map<(d0, d1) -> ((d0 * 72 + d1) floordiv 2304)>
#map4 = affine_map<(d0, d1) -> (((d0 * 72 + d1) mod 2304) floordiv 1152)>
#map5 = affine_map<(d0, d1) -> (((((d0 * 72 + d1) mod 2304) mod 1152) floordiv 9) floordiv 8)>
#map6 = affine_map<(d0, d1) -> (((((d0 * 72 + d1) mod 2304) mod 1152) mod 9) floordiv 3)>
#map7 = affine_map<(d0, d1) -> (((((d0 * 72 + d1) mod 2304) mod 1152) mod 9) mod 3)>
#map10 = affine_map<(d0, d1) -> (d0 * 16 + d1)>
#map11 = affine_map<(d0, d1) -> (d0 * 16 + d1)>
#map12 = affine_map<(d0, d1) -> (d0 * 16 - d1 + 15)>
func @fuse_across_varying_dims_complex(%arg0: f32) {
  %c0 = constant 0 : index
  %0 = memref.alloc() : memref<2x2x3x3x16x1xf32>
  %1 = memref.alloc() : memref<64x9xf32>
  %2 = memref.alloc() : memref<144x4xf32>
  affine.for %i0 = 0 to 64 {
    affine.for %i1 = 0 to 9 {
      %4 = affine.apply #map3(%i0, %i1)
      %5 = affine.apply #map4(%i0, %i1)
      %6 = affine.apply #map5(%i0, %i1)
      %7 = affine.apply #map6(%i0, %i1)
      %8 = affine.apply #map7(%i0, %i1)
      %9 = affine.load %0[%4, %5, %7, %8, %6, %c0] : memref<2x2x3x3x16x1xf32>
      affine.store %9, %1[%i0, %i1] : memref<64x9xf32>
    }
  }
  affine.for %i2 = 0 to 9 {
    affine.for %i3 = 0 to 4 {
      affine.for %i4 = 0 to 16 {
        %10 = affine.apply #map10(%i3, %i4)
        %11 = affine.load %1[%10, %i2] : memref<64x9xf32>
      }
      affine.for %i5 = 0 to 16 {
        %14 = affine.apply #map11(%i2, %i5)
        affine.store %arg0, %2[%14, %i3] : memref<144x4xf32>
      }
    }
  }
  affine.for %i6 = 0 to 9 {
    affine.for %i7 = 0 to 9 {
      affine.for %i8 = 0 to 4 {
        affine.for %i9 = 0 to 16 {
          %15 = affine.apply #map12(%i8, %i9)
          %16 = affine.load %1[%15, %i7] : memref<64x9xf32>
        }
      }
    }
  }
  return
}
// MAXIMAL-DAG: [[$MAP0:#map[0-9]+]] = affine_map<(d0, d1) -> ((d0 * 72 + d1) floordiv 2304)>
// MAXIMAL-DAG: [[$MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (((d0 * 72 + d1) mod 2304) floordiv 1152)>
// MAXIMAL-DAG: [[$MAP2:#map[0-9]+]] = affine_map<(d0, d1) -> (((((d0 * 72 + d1) mod 2304) mod 1152) floordiv 9) floordiv 8)>
// MAXIMAL-DAG: [[$MAP3:#map[0-9]+]] = affine_map<(d0, d1) -> (((((d0 * 72 + d1) mod 2304) mod 1152) mod 9) floordiv 3)>
// MAXIMAL-DAG: [[$MAP4:#map[0-9]+]] = affine_map<(d0, d1) -> (((((d0 * 72 + d1) mod 2304) mod 1152) mod 9) mod 3)>
// MAXIMAL-DAG: [[$MAP7:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 * 16 + d1)>
// MAXIMAL-DAG: [[$MAP8:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 * 16 - d1 + 15)>
// MAXIMAL-LABEL: func @fuse_across_varying_dims_complex
// MAXIMAL-NEXT:  memref.alloc() : memref<64x1xf32>
// MAXIMAL-NEXT:  constant 0 : index
// MAXIMAL-NEXT:  memref.alloc() : memref<2x2x3x3x16x1xf32>
// MAXIMAL-NEXT:  memref.alloc() : memref<144x4xf32>
// MAXIMAL-NEXT:  affine.for %{{.*}} = 0 to 9 {
// MAXIMAL-NEXT:    affine.for %{{.*}} = 0 to 9 {
// MAXIMAL-NEXT:      affine.for %{{.*}} = 0 to 4 {
// MAXIMAL-NEXT:        affine.for %{{.*}} = 0 to 16 {
// MAXIMAL-NEXT:          affine.for %{{.*}} = 0 to 64 {
// MAXIMAL-NEXT:            affine.apply [[$MAP0]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:            affine.apply [[$MAP1]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:            affine.apply [[$MAP2]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:            affine.apply [[$MAP3]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:            affine.apply [[$MAP4]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:            affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<2x2x3x3x16x1xf32>
// MAXIMAL-NEXT:            affine.store %{{.*}}, %{{.*}}[%{{.*}}, 0] : memref<64x1xf32>
// MAXIMAL-NEXT:          }
// MAXIMAL-NEXT:          affine.for %{{.*}} = 0 to 4 {
// MAXIMAL-NEXT:            affine.for %{{.*}} = 0 to 16 {
// MAXIMAL-NEXT:              affine.apply [[$MAP7]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:              affine.load %{{.*}}[%{{.*}} * 16 + %{{.*}}, 0] : memref<64x1xf32>
// MAXIMAL-NEXT:            }
// MAXIMAL-NEXT:            affine.for %{{.*}} = 0 to 16 {
// MAXIMAL-NEXT:              affine.apply [[$MAP7]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:              affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<144x4xf32>
// MAXIMAL-NEXT:            }
// MAXIMAL-NEXT:          }
// MAXIMAL-NEXT:          affine.apply [[$MAP8]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:          affine.load %{{.*}}[%{{.*}} * 16 - %{{.*}} + 15, 0] : memref<64x1xf32>
// MAXIMAL-NEXT:        }
// MAXIMAL-NEXT:      }
// MAXIMAL-NEXT:    }
// MAXIMAL-NEXT:  }

// -----

func @should_fuse_with_slice_union() {
  %a = memref.alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32

  affine.for %i0 = 0 to 100 {
    affine.store %cf0, %a[%i0]: memref<100xf32>
  }

  affine.for %i1 = 10 to 20 {
    %v0 = affine.load %a[%i1]: memref<100xf32>
    affine.for %i2 = 15 to 25 {
      %v1 = affine.load %a[%i2]: memref<100xf32>
    }
  }
  // The union of two slice bounds (calculated between the store and each of
  // the loads) is computed and used in the fusion cost calculation, index
  // remapping, and private memref size. The result is that the temporary
  // memref is reduced from 100xf32 to 15xf32 and properly indexed by
  // the fused loops based on the union calculation.
// CHECK:      affine.for %{{.*}} = 10 to 20 {
// CHECK-NEXT:   affine.for %{{.*}} = 10 to 25 {
// CHECK-NEXT:     affine.store %{{.*}}, %{{.*}}[%{{.*}} - 10] : memref<15xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.load %{{.*}}[%{{.*}} - 10] : memref<15xf32>
// CHECK-NEXT:   affine.for %{{.*}} = 15 to 25 {
// CHECK-NEXT:     affine.load %{{.*}}[%{{.*}} - 10] : memref<15xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: return
  return
}

// -----

func @affine_add_mm_fused(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>, %arg3: memref<1024x1024xf32>) {
  affine.for %i2 = 0 to 1024 {
    affine.for %i3 = 0 to 1024 {
      %0 = affine.load %arg3[%i2, %i3] : memref<1024x1024xf32>
      %1 = affine.load %arg2[%i2, %i3] : memref<1024x1024xf32>
      %2 = addf %1, %0 : f32
      affine.store %2, %arg2[%i2, %i3] : memref<1024x1024xf32>
    }
  }
  affine.for %i4 = 0 to 1024 {
    affine.for %i5 = 0 to 1024 {
      affine.for %i6 = 0 to 1024 {
        %3 = affine.load %arg1[%i6, %i5] : memref<1024x1024xf32>
        %4 = affine.load %arg0[%i4, %i6] : memref<1024x1024xf32>
        %5 = mulf %4, %3 : f32
        %6 = affine.load %arg2[%i4, %i5] : memref<1024x1024xf32>
        %7 = addf %6, %5 : f32
        affine.store %7, %arg2[%i4, %i5] : memref<1024x1024xf32>
      }
    }
  }
  // Should fuse elementwise add loop at loop depth 2, above loop-carried
  // dependence between load/store on '%arg2', carried on reduction loop %i6.
  // CHECK:       affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:      addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:      affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:        affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:        affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:        mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:        affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:        addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:        affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  return
}

// -----

func @affine_2mm_fused(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>, %arg3: memref<1024x1024xf32>, %arg4: memref<1024x1024xf32>) {
  %cst = constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 1024 {
    affine.for %i1 = 0 to 1024 {
      affine.store %cst, %arg2[%i0, %i1] : memref<1024x1024xf32>
    }
  }
  affine.for %i2 = 0 to 1024 {
    affine.for %i3 = 0 to 1024 {
      affine.store %cst, %arg4[%i2, %i3] : memref<1024x1024xf32>
    }
  }
  affine.for %i4 = 0 to 1024 {
    affine.for %i5 = 0 to 1024 {
      affine.for %i6 = 0 to 1024 {
        %0 = affine.load %arg1[%i6, %i5] : memref<1024x1024xf32>
        %1 = affine.load %arg0[%i4, %i6] : memref<1024x1024xf32>
        %2 = mulf %1, %0 : f32
        %3 = affine.load %arg2[%i4, %i5] : memref<1024x1024xf32>
        %4 = addf %3, %2 : f32
        affine.store %4, %arg2[%i4, %i5] : memref<1024x1024xf32>
      }
    }
  }
  affine.for %i7 = 0 to 1024 {
    affine.for %i8 = 0 to 1024 {
      affine.for %i9 = 0 to 1024 {
        %5 = affine.load %arg1[%i9, %i8] : memref<1024x1024xf32>
        %6 = affine.load %arg0[%i7, %i9] : memref<1024x1024xf32>
        %7 = mulf %6, %5 : f32
        %8 = affine.load %arg4[%i7, %i8] : memref<1024x1024xf32>
        %9 = addf %8, %7 : f32
        affine.store %9, %arg4[%i7, %i8] : memref<1024x1024xf32>
      }
    }
  }

  // Should fuse MM initialization loops into their consumers, then fuse the
  // two matmul loops together for input reuse on '%arg0/%arg1'.

  // CHECK:        affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:     affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }

  return
}

// -----

func @affine_2_dependent_mm_fused(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>, %arg3: memref<1024x1024xf32>, %arg4: memref<1024x1024xf32>) {
  affine.for %i0 = 0 to 1024 {
    affine.for %i1 = 0 to 1024 {
      affine.for %i2 = 0 to 1024 {
        %0 = affine.load %arg1[%i2, %i1] : memref<1024x1024xf32>
        %1 = affine.load %arg0[%i0, %i2] : memref<1024x1024xf32>
        %2 = mulf %1, %0 : f32
        %3 = affine.load %arg2[%i0, %i1] : memref<1024x1024xf32>
        %4 = addf %3, %2 : f32
        affine.store %4, %arg2[%i0, %i1] : memref<1024x1024xf32>
      }
    }
  }
  affine.for %i3 = 0 to 1024 {
    affine.for %i4 = 0 to 1024 {
      affine.for %i5 = 0 to 1024 {
        %5 = affine.load %arg3[%i5, %i4] : memref<1024x1024xf32>
        %6 = affine.load %arg2[%i3, %i5] : memref<1024x1024xf32>
        %7 = mulf %6, %5 : f32
        %8 = affine.load %arg4[%i3, %i4] : memref<1024x1024xf32>
        %9 = addf %8, %7 : f32
        affine.store %9, %arg4[%i3, %i4] : memref<1024x1024xf32>
      }
    }
  }

  // CHECK:        affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:     affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:       affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:       affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  return
}

// -----

// CHECK-LABEL: func @should_fuse_self_dependence_multi_store_producer() {
func @should_fuse_self_dependence_multi_store_producer() {
  %m = memref.alloc() : memref<10xf32>
  %local_m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %local_m[%i0] : memref<10xf32>
    %v0 = affine.load %local_m[%i0] : memref<10xf32>
    affine.store %v0, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v1 = affine.load %m[%i1] : memref<10xf32>
  }
  // CHECK:      affine.for %[[i0:.*]] = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, [[LOCAL_M:%.*]][%[[i0]]] : memref<10xf32>
  // CHECK-NEXT:   [[v0:%.*]] = affine.load [[LOCAL_M]][%[[i0]]] : memref<10xf32>
  // CHECK-NEXT:   affine.store [[v0]], %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_dead_multi_store_producer() {
func @should_fuse_dead_multi_store_producer() {
  %m = memref.alloc() : memref<10xf32>
  %dead_m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %dead_m[%i0] : memref<10xf32>
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }
  // CHECK:      affine.for %[[i0:.*]] = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%[[i0]]] : memref<10xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_function_live_out_multi_store_producer
func @should_fuse_function_live_out_multi_store_producer(%live_in_out_m : memref<10xf32>) {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %live_in_out_m[%i0] : memref<10xf32>
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }
  // CHECK:      affine.for %[[i0:.*]] = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%[[i0]]] : memref<10xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// Test case from github bug 777.
// CHECK-LABEL: func @mul_add_0
func @mul_add_0(%arg0: memref<3x4xf32>, %arg1: memref<4x3xf32>, %arg2: memref<3x3xf32>, %arg3: memref<3x3xf32>) {
  %cst = constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<3x3xf32>
  affine.for %arg4 = 0 to 3 {
    affine.for %arg5 = 0 to 3 {
      affine.store %cst, %0[%arg4, %arg5] : memref<3x3xf32>
    }
  }
  affine.for %arg4 = 0 to 3 {
    affine.for %arg5 = 0 to 3 {
      affine.for %arg6 = 0 to 4 {
        %1 = affine.load %arg1[%arg6, %arg5] : memref<4x3xf32>
        %2 = affine.load %arg0[%arg4, %arg6] : memref<3x4xf32>
        %3 = mulf %2, %1 : f32
        %4 = affine.load %0[%arg4, %arg5] : memref<3x3xf32>
        %5 = addf %4, %3 : f32
        affine.store %5, %0[%arg4, %arg5] : memref<3x3xf32>
      }
    }
  }
  affine.for %arg4 = 0 to 3 {
    affine.for %arg5 = 0 to 3 {
      %6 = affine.load %arg2[%arg4, %arg5] : memref<3x3xf32>
      %7 = affine.load %0[%arg4, %arg5] : memref<3x3xf32>
      %8 = addf %7, %6 : f32
      affine.store %8, %arg3[%arg4, %arg5] : memref<3x3xf32>
    }
  }
  // CHECK:      affine.for %[[i0:.*]] = 0 to 3 {
  // CHECK-NEXT:   affine.for %[[i1:.*]] = 0 to 3 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:     affine.for %[[i2:.*]] = 0 to 4 {
  // CHECK-NEXT:       affine.load %{{.*}}[%[[i2]], %[[i1]]] : memref<4x3xf32>
  // CHECK-NEXT:       affine.load %{{.*}}[%[[i0]], %[[i2]]] : memref<3x4xf32>
  // CHECK-NEXT:       mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:       affine.load %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:       addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:     }
  // CHECK-NEXT:     affine.load %{{.*}}[%[[i0]], %[[i1]]] : memref<3x3xf32>
  // CHECK-NEXT:     affine.load %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:     addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:     affine.store %{{.*}}, %{{.*}}[%[[i0]], %[[i1]]] : memref<3x3xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// Verify that 'fuseProducerConsumerNodes' fuse a producer loop with a store
// that has multiple outgoing edges.

// CHECK-LABEL: func @should_fuse_multi_outgoing_edge_store_producer
func @should_fuse_multi_outgoing_edge_store_producer(%a : memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  affine.for %arg0 = 0 to 1 {
    affine.store %cst, %a[%arg0] : memref<1xf32>
  }

  affine.for %arg0 = 0 to 1 {
    %0 = affine.load %a[%arg0] : memref<1xf32>
  }

  affine.for %arg0 = 0 to 1 {
    %0 = affine.load %a[%arg0] : memref<1xf32>
  }
  // CHECK:      affine.for %{{.*}} = 0 to 1 {
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT: }

  return
}

// -----

// Verify that 'fuseProducerConsumerNodes' fuses a producer loop that: 1) has
// multiple outgoing edges, 2) producer store has a single outgoing edge.
// Sibling loop fusion should not fuse any of these loops due to
// dependencies on external memrefs '%a' and '%b'.

// CHECK-LABEL: func @should_fuse_producer_with_multi_outgoing_edges
func @should_fuse_producer_with_multi_outgoing_edges(%a : memref<1xf32>, %b : memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  affine.for %arg0 = 0 to 1 {
    %0 = affine.load %a[%arg0] : memref<1xf32>
    affine.store %cst, %b[%arg0] : memref<1xf32>
  }

  affine.for %arg0 = 0 to 1 {
    affine.store %cst, %a[%arg0] : memref<1xf32>
    %1 = affine.load %b[%arg0] : memref<1xf32>
  }
  // CHECK: affine.for %{{.*}} = 0 to 1
  // CHECK-NEXT: affine.load %[[A:.*]][{{.*}}]
  // CHECK-NEXT: affine.store %{{.*}}, %[[B:.*]][{{.*}}]
  // CHECK-NEXT: affine.store %{{.*}}, %[[A]]
  // CHECK-NEXT: affine.load %[[B]]
  // CHECK-NOT: affine.for %{{.*}}
  // CHECK: return
  return
}

// -----

// MAXIMAL-LABEL: func @reshape_into_matmul
func @reshape_into_matmul(%lhs : memref<1024x1024xf32>,
              %R: memref<16x64x1024xf32>, %out: memref<1024x1024xf32>) {
  %rhs = memref.alloc() :  memref<1024x1024xf32>

  // Reshape from 3-d to 2-d.
  affine.for %i0 = 0 to 16 {
    affine.for %i1 = 0 to 64 {
      affine.for %k = 0 to 1024 {
        %v = affine.load %R[%i0, %i1, %k] : memref<16x64x1024xf32>
        affine.store %v, %rhs[64*%i0 + %i1, %k] : memref<1024x1024xf32>
      }
    }
  }

  // Matmul.
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %0 = affine.load %rhs[%k, %j] : memref<1024x1024xf32>
        %1 = affine.load %lhs[%i, %k] : memref<1024x1024xf32>
        %2 = mulf %1, %0 : f32
        %3 = affine.load %out[%i, %j] : memref<1024x1024xf32>
        %4 = addf %3, %2 : f32
        affine.store %4, %out[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return
}
// MAXIMAL-NEXT: memref.alloc
// MAXIMAL-NEXT: affine.for
// MAXIMAL-NEXT:   affine.for
// MAXIMAL-NEXT:     affine.for
// MAXIMAL-NOT:      affine.for
// MAXIMAL:      return

// -----

// CHECK-LABEL: func @vector_loop
func @vector_loop(%a : memref<10x20xf32>, %b : memref<10x20xf32>,
                  %c : memref<10x20xf32>) {
  affine.for %j = 0 to 10 {
    affine.for %i = 0 to 5 {
      %ld0 = affine.vector_load %a[%j, %i*4] : memref<10x20xf32>, vector<4xf32>
      affine.vector_store %ld0, %b[%j, %i*4] : memref<10x20xf32>, vector<4xf32>
    }
  }

  affine.for %j = 0 to 10 {
    affine.for %i = 0 to 5 {
      %ld0 = affine.vector_load %b[%j, %i*4] : memref<10x20xf32>, vector<4xf32>
      affine.vector_store %ld0, %c[%j, %i*4] : memref<10x20xf32>, vector<4xf32>
    }
  }

  return
}
// CHECK:      affine.for
// CHECK-NEXT:   affine.for
// CHECK-NEXT:     affine.vector_load
// CHECK-NEXT:     affine.vector_store
// CHECK-NEXT:     affine.vector_load
// CHECK-NEXT:     affine.vector_store
// CHECK-NOT:  affine.for

// -----

// CHECK-LABEL: func @multi_outgoing_edges
func @multi_outgoing_edges(%in0 : memref<32xf32>,
                      %in1 : memref<32xf32>) {
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = addf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = subf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = mulf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = divf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  return
}

// CHECK:      affine.for
// CHECK-NOT:  affine.for
// CHECK:        addf
// CHECK-NOT:  affine.for
// CHECK:        subf
// CHECK-NOT:  affine.for
// CHECK:        mulf
// CHECK-NOT:  affine.for
// CHECK:        divf

// -----

// Test fusion when dynamically shaped memrefs are used with constant trip count loops.

// CHECK-LABEL: func @calc
func @calc(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %len: index) {
  %c1 = constant 1 : index
  %1 = memref.alloc(%len) : memref<?xf32>
  affine.for %arg4 = 1 to 10 {
    %7 = affine.load %arg0[%arg4] : memref<?xf32>
    %8 = affine.load %arg1[%arg4] : memref<?xf32>
    %9 = addf %7, %8 : f32
    affine.store %9, %1[%arg4] : memref<?xf32>
  }
  affine.for %arg4 = 1 to 10 {
    %7 = affine.load %1[%arg4] : memref<?xf32>
    %8 = affine.load %arg1[%arg4] : memref<?xf32>
    %9 = mulf %7, %8 : f32
    affine.store %9, %arg2[%arg4] : memref<?xf32>
  }
  return
}
// CHECK:       memref.alloc() : memref<1xf32>
// CHECK:       affine.for %arg{{.*}} = 1 to 10 {
// CHECK-NEXT:    affine.load %arg{{.*}}
// CHECK-NEXT:    affine.load %arg{{.*}}
// CHECK-NEXT:    addf
// CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
// CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
// CHECK-NEXT:    affine.load %arg{{.*}}[%arg{{.*}}] : memref<?xf32>
// CHECK-NEXT:    mulf
// CHECK-NEXT:    affine.store %{{.*}}, %arg{{.*}}[%arg{{.*}}] : memref<?xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// -----

// CHECK-LABEL: func @should_not_fuse_since_non_affine_users
func @should_not_fuse_since_non_affine_users(%in0 : memref<32xf32>,
                      %in1 : memref<32xf32>) {
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = addf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  affine.for %d = 0 to 32 {
    %lhs = memref.load %in0[%d] : memref<32xf32>
    %rhs = memref.load %in1[%d] : memref<32xf32>
    %add = subf %lhs, %rhs : f32
    memref.store %add, %in0[%d] : memref<32xf32>
  }
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = mulf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  return
}

// CHECK:  affine.for
// CHECK:    addf
// CHECK:  affine.for
// CHECK:    subf
// CHECK:  affine.for
// CHECK:    mulf

// -----

// CHECK-LABEL: func @should_not_fuse_since_top_level_non_affine_users
func @should_not_fuse_since_top_level_non_affine_users(%in0 : memref<32xf32>,
                      %in1 : memref<32xf32>) {
  %sum = memref.alloc() : memref<f32>
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = addf %lhs, %rhs : f32
    memref.store %add, %sum[] : memref<f32>
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  %load_sum = memref.load %sum[] : memref<f32>
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = mulf %lhs, %rhs : f32
    %sub = subf %add, %load_sum: f32
    affine.store %sub, %in0[%d] : memref<32xf32>
  }
  memref.dealloc %sum : memref<f32>
  return
}

// CHECK:  affine.for
// CHECK:    addf
// CHECK:  affine.for
// CHECK:    mulf
// CHECK:    subf

// -----

// CHECK-LABEL: func @should_not_fuse_since_top_level_non_affine_mem_write_users
func @should_not_fuse_since_top_level_non_affine_mem_write_users(
    %in0 : memref<32xf32>, %in1 : memref<32xf32>) {
  %c0 = constant 0 : index
  %cst_0 = constant 0.000000e+00 : f32

  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = addf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  memref.store %cst_0, %in0[%c0] : memref<32xf32>
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = addf %lhs, %rhs: f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  return
}

// CHECK:  affine.for
// CHECK:    addf
// CHECK:  affine.for
// CHECK:    addf

// -----

// MAXIMAL-LABEL: func @fuse_minor_affine_map
func @fuse_minor_affine_map(%in: memref<128xf32>, %out: memref<20x512xf32>) {
  %tmp = memref.alloc() : memref<128xf32>

  affine.for %arg4 = 0 to 128 {
    %ld = affine.load %in[%arg4] : memref<128xf32>
    affine.store %ld, %tmp[%arg4] : memref<128xf32>
  }

  affine.for %arg3 = 0 to 20 {
    affine.for %arg4 = 0 to 512 {
      %ld = affine.load %tmp[%arg4 mod 128] : memref<128xf32>
      affine.store %ld, %out[%arg3, %arg4] : memref<20x512xf32>
    }
  }

  return
}

// TODO: The size of the private memref is not properly computed in the presence
// of the 'mod' operation. It should be memref<1xf32> instead of
// memref<128xf32>: https://bugs.llvm.org/show_bug.cgi?id=46973
// MAXIMAL:       memref.alloc() : memref<128xf32>
// MAXIMAL:       affine.for
// MAXIMAL-NEXT:    affine.for
// MAXIMAL-NOT:   affine.for
// MAXIMAL:       return

// -----

// CHECK-LABEL: func @should_fuse_multi_store_producer_and_privatize_memfefs
func @should_fuse_multi_store_producer_and_privatize_memfefs() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>
  %cst = constant 0.000000e+00 : f32
  affine.for %arg0 = 0 to 10 {
    affine.store %cst, %a[%arg0] : memref<10xf32>
    affine.store %cst, %b[%arg0] : memref<10xf32>
    affine.store %cst, %c[%arg0] : memref<10xf32>
    %0 = affine.load %c[%arg0] : memref<10xf32>
  }

  affine.for %arg0 = 0 to 10 {
    %0 = affine.load %a[%arg0] : memref<10xf32>
  }

  affine.for %arg0 = 0 to 10 {
    %0 = affine.load %b[%arg0] : memref<10xf32>
  }

	// All the memrefs should be privatized except '%c', which is not involved in
  // the producer-consumer fusion.
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }

  return
}

// -----

func @should_fuse_multi_store_producer_with_scaping_memrefs_and_remove_src(
    %a : memref<10xf32>, %b : memref<10xf32>) {
  %cst = constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %a[%i0] : memref<10xf32>
    affine.store %cst, %b[%i0] : memref<10xf32>
  }

  affine.for %i1 = 0 to 10 {
    %0 = affine.load %a[%i1] : memref<10xf32>
  }

  affine.for %i2 = 0 to 10 {
    %0 = affine.load %b[%i2] : memref<10xf32>
  }

	// Producer loop '%i0' should be removed after fusion since fusion is maximal.
  // No memref should be privatized since they escape the function.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NOT:   affine.for

  return
}

// -----

func @should_fuse_multi_store_producer_with_scaping_memrefs_and_preserve_src(
    %a : memref<10xf32>, %b : memref<10xf32>) {
  %cst = constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %a[%i0] : memref<10xf32>
    affine.store %cst, %b[%i0] : memref<10xf32>
  }

  affine.for %i1 = 0 to 5 {
    %0 = affine.load %a[%i1] : memref<10xf32>
  }

  affine.for %i2 = 0 to 10 {
    %0 = affine.load %b[%i2] : memref<10xf32>
  }

	// Loops '%i0' and '%i2' should be fused first and '%i0' should be removed
  // since fusion is maximal. Then the fused loop and '%i1' should be fused
  // and the fused loop shouldn't be removed since fusion is not maximal.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK:       affine.for %{{.*}} = 0 to 5 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NOT:   affine.for

  return
}

// -----

func @should_not_fuse_due_to_dealloc(%arg0: memref<16xf32>){
  %A = memref.alloc() : memref<16xf32>
  %C = memref.alloc() : memref<16xf32>
  %cst_1 = constant 1.000000e+00 : f32
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %arg0[%arg1] : memref<16xf32>
    affine.store %a, %A[%arg1] : memref<16xf32>
    affine.store %a, %C[%arg1] : memref<16xf32>
  }
  memref.dealloc %C : memref<16xf32>
  %B = memref.alloc() : memref<16xf32>
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %A[%arg1] : memref<16xf32>
    %b = addf %cst_1, %a : f32
    affine.store %b, %B[%arg1] : memref<16xf32>
  }
  memref.dealloc %A : memref<16xf32>
  return
}
// CHECK-LABEL: func @should_not_fuse_due_to_dealloc
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      affine.store
// CHECK-NEXT:      affine.store
// CHECK:         memref.dealloc
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      addf
// CHECK-NEXT:      affine.store

// -----

// CHECK-LABEL: func @should_fuse_defining_node_has_no_dependence_from_source_node
func @should_fuse_defining_node_has_no_dependence_from_source_node(
    %a : memref<10xf32>, %b : memref<f32>) -> () {
  affine.for %i0 = 0 to 10 {
    %0 = affine.load %b[] : memref<f32>
    affine.store %0, %a[%i0] : memref<10xf32>
  }
  %0 = affine.load %b[] : memref<f32>
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %a[%i1] : memref<10xf32>
    %2 = divf %0, %1 : f32
  }

	// Loops '%i0' and '%i1' should be fused even though there is a defining
  // node between the loops. It is because the node has no dependence from '%i0'.
  // CHECK:       affine.load %{{.*}}[] : memref<f32>
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[] : memref<f32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    divf
  // CHECK-NEXT:  }
  // CHECK-NOT:   affine.for
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_defining_node_has_dependence_from_source_loop
func @should_not_fuse_defining_node_has_dependence_from_source_loop(
    %a : memref<10xf32>, %b : memref<f32>) -> () {
  %cst = constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %b[] : memref<f32>
    affine.store %cst, %a[%i0] : memref<10xf32>
  }
  %0 = affine.load %b[] : memref<f32>
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %a[%i1] : memref<10xf32>
    %2 = divf %0, %1 : f32
  }

	// Loops '%i0' and '%i1' should not be fused because the defining node
  // of '%0' used in '%i1' has dependence from loop '%i0'.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[] : memref<f32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.load %{{.*}}[] : memref<f32>
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    divf
  // CHECK-NEXT:  }
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_defining_node_has_transitive_dependence_from_source_loop
func @should_not_fuse_defining_node_has_transitive_dependence_from_source_loop(
    %a : memref<10xf32>, %b : memref<10xf32>, %c : memref<f32>) -> () {
  %cst = constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %a[%i0] : memref<10xf32>
    affine.store %cst, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %b[%i1] : memref<10xf32>
    affine.store %1, %c[] : memref<f32>
  }
  %0 = affine.load %c[] : memref<f32>
  affine.for %i2 = 0 to 10 {
    %1 = affine.load %a[%i2] : memref<10xf32>
    %2 = divf %0, %1 : f32
  }

	// When loops '%i0' and '%i2' are evaluated first, they should not be
  // fused. The defining node of '%0' in loop '%i2' has transitive dependence
  // from loop '%i0'. After that, loops '%i0' and '%i1' are evaluated, and they
  // will be fused as usual.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[] : memref<f32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.load %{{.*}}[] : memref<f32>
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    divf
  // CHECK-NEXT:  }
  // CHECK-NOT:   affine.for
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_dest_loop_nest_return_value
func @should_not_fuse_dest_loop_nest_return_value(
    %a : memref<10xf32>) -> () {
  %cst = constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %a[%i0] : memref<10xf32>
  }
  %b = affine.for %i1 = 0 to 10 step 2 iter_args(%b_iter = %cst) -> f32 {
    %load_a = affine.load %a[%i1] : memref<10xf32>
    affine.yield %load_a: f32
  }

  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK:       affine.for %{{.*}} = 0 to 10 step 2 iter_args(%{{.*}} = %{{.*}}) -> (f32) {
  // CHECK-NEXT:    affine.load
  // CHECK-NEXT:    affine.yield
  // CHECK-NEXT:  }

  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_src_loop_nest_return_value
func @should_not_fuse_src_loop_nest_return_value(
    %a : memref<10xf32>) -> () {
  %cst = constant 1.000000e+00 : f32
  %b = affine.for %i = 0 to 10 step 2 iter_args(%b_iter = %cst) -> f32 {
    %c = addf %b_iter, %b_iter : f32
    affine.store %c, %a[%i] : memref<10xf32>
    affine.yield %c: f32
  }
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %a[%i1] : memref<10xf32>
  }

  // CHECK:       %{{.*}} = affine.for %{{.*}} = 0 to 10 step 2 iter_args(%{{.*}} = %{{.*}}) -> (f32) {
  // CHECK-NEXT:    %{{.*}} = addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.yield %{{.*}} : f32
  // CHECK-NEXT:  }
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }

  return
}

// -----

func private @some_function(memref<16xf32>)
func @call_op_prevents_fusion(%arg0: memref<16xf32>){
  %A = memref.alloc() : memref<16xf32>
  %cst_1 = constant 1.000000e+00 : f32
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %arg0[%arg1] : memref<16xf32>
    affine.store %a, %A[%arg1] : memref<16xf32>
  }
  call @some_function(%A) : (memref<16xf32>) -> ()
  %B = memref.alloc() : memref<16xf32>
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %A[%arg1] : memref<16xf32>
    %b = addf %cst_1, %a : f32
    affine.store %b, %B[%arg1] : memref<16xf32>
  }
  return
}
// CHECK-LABEL: func @call_op_prevents_fusion
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      affine.store
// CHECK:         call
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      addf
// CHECK-NEXT:      affine.store

// -----

func private @some_function()
func @call_op_does_not_prevent_fusion(%arg0: memref<16xf32>){
  %A = memref.alloc() : memref<16xf32>
  %cst_1 = constant 1.000000e+00 : f32
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %arg0[%arg1] : memref<16xf32>
    affine.store %a, %A[%arg1] : memref<16xf32>
  }
  call @some_function() : () -> ()
  %B = memref.alloc() : memref<16xf32>
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %A[%arg1] : memref<16xf32>
    %b = addf %cst_1, %a : f32
    affine.store %b, %B[%arg1] : memref<16xf32>
  }
  return
}
// CHECK-LABEL: func @call_op_does_not_prevent_fusion
// CHECK:         affine.for
// CHECK-NOT:     affine.for

// -----

// Fusion is avoided when the slice computed is invalid. Comments below describe
// incorrect backward slice computation. Similar logic applies for forward slice
// as well.
func @no_fusion_cannot_compute_valid_slice() {
  %A = memref.alloc() : memref<5xf32>
  %B = memref.alloc() : memref<6xf32>
  %C = memref.alloc() : memref<5xf32>
  %cst = constant 0. : f32

  affine.for %arg0 = 0 to 5 {
    %a = affine.load %A[%arg0] : memref<5xf32>
    affine.store %a, %B[%arg0 + 1] : memref<6xf32>
  }

  affine.for %arg0 = 0 to 5 {
    // Backward slice computed will be:
    // slice ( src loop: 0, dst loop: 1, depth: 1 : insert point: (1, 0)
    // loop bounds: [(d0) -> (d0 - 1), (d0) -> (d0)] )

    // Resulting fusion would be as below. It is easy to note the out-of-bounds
    // access by 'affine.load'.

    // #map0 = affine_map<(d0) -> (d0 - 1)>
    // #map1 = affine_map<(d0) -> (d0)>
    // affine.for %arg1 = #map0(%arg0) to #map1(%arg0) {
    //   %5 = affine.load %1[%arg1] : memref<5xf32>
    //   ...
    //   ...
    // }

    %a = affine.load %B[%arg0] : memref<6xf32>
    %b = mulf %a, %cst : f32
    affine.store %b, %C[%arg0] : memref<5xf32>
  }
  return
}
// CHECK-LABEL: func @no_fusion_cannot_compute_valid_slice
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      affine.store
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      mulf
// CHECK-NEXT:      affine.store

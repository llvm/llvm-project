// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion))' -split-input-file | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{fusion-maximal}))' -split-input-file | FileCheck %s --check-prefix=MAXIMAL

// Part I of fusion tests in  mlir/test/Transforms/loop-fusion.mlir.
// Part III of fusion tests in mlir/test/Transforms/loop-fusion-3.mlir
// Part IV of fusion tests in mlir/test/Transforms/loop-fusion-4.mlir

// -----

// CHECK-LABEL: func @should_fuse_at_depth_above_loop_carried_dependence(%{{.*}}: memref<64x4xf32>, %{{.*}}: memref<64x4xf32>) {
func.func @should_fuse_at_depth_above_loop_carried_dependence(%arg0: memref<64x4xf32>, %arg1: memref<64x4xf32>) {
  %out = memref.alloc() : memref<64x4xf32>
  %0 = arith.constant 0.0 : f32
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
          %s = arith.addf %v, %r : f32
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
  // CHECK-NEXT:          arith.addf %{{.*}}, %{{.*}} : f32
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
func.func @should_fuse_only_two_loops_and_remove_producer() {
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
func.func @should_fuse_after_one_loop_interchange() {
  %a = memref.alloc() : memref<10xf32>

  %cf0 = arith.constant 0.0 : f32
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
func.func @should_fuse_after_two_loop_interchanges() {
  %a = memref.alloc() : memref<6x8xf32>

  %cf0 = arith.constant 0.0 : f32
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
          %v1 = arith.addf %v0, %v0 : f32
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
  // CHECK-NEXT:          arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:          affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

func.func @should_fuse_live_out_writer(%arg0 : memref<10xf32>) -> memref<10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %arg0[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %arg0[%i1] : memref<10xf32>
    affine.store %1, %arg0[%i1] : memref<10xf32>
  }
  return %arg0 : memref<10xf32>

  // CHECK:       %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return %{{.*}} : memref<10xf32>
}

// -----

// The fused slice has 16 iterations from along %i0.

// CHECK-DAG: [[$MAP_LB:#map[0-9]*]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG: [[$MAP_UB:#map[0-9]*]] = affine_map<(d0) -> (d0 * 16 + 16)>

// CHECK-LABEL: slice_tile
func.func @slice_tile(%arg0: memref<128x8xf32>, %arg1: memref<32x8xf32>, %0 : f32) -> memref<32x8xf32> {
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
          %s = arith.addf %v, %v : f32
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
// CHECK-NEXT:          arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:          affine.store %{{.*}}, %{{.*}}[%{{.*}} * 16 + %{{.*}}, %{{.*}}] : memref<32x8xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return %{{.*}} : memref<32x8xf32>
// CHECK-NEXT:}

// -----

// Test case which illustrates fix for b/126454413
func.func @test_add_slice_bounds() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %c0 = arith.constant 0 : index

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
// CHECK-NEXT:         affine.apply #map(%{{.*}})
// CHECK-NEXT:         affine.apply #map(%{{.*}})
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

func.func @should_fuse_init_loops_siblings_then_shared_producer(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) {
  %0 = memref.alloc() : memref<10x10xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 7.000000e+00 : f32
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
      %3 = arith.mulf %1, %2 : f32
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
      %6 = arith.addf %4, %5 : f32
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
// CHECK-NEXT:       arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       affine.load %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return

  return
}

// -----

// Test sibling fusion of two matrix-vector products sharing the input matrix.

func.func @two_matrix_vector_products() {
  %in_matrix = memref.alloc() : memref<10x10xf32>
  %in_vec0 = memref.alloc() : memref<10xf32>
  %in_vec1 = memref.alloc() : memref<10xf32>
  %out_vec0 = memref.alloc() : memref<10xf32>
  %out_vec1 = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

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
      %v2 = arith.mulf %v0, %v1 : f32
      %v3 = affine.load %out_vec0[%i3] : memref<10xf32>
      %v4 = arith.addf %v2, %v3 : f32
      affine.store %v4, %out_vec0[%i3] : memref<10xf32>
    }
  }
  // out_vec1 = in_matrix x in_vec1
  affine.for %i4 = 0 to 10 {
    affine.for %i5 = 0 to 10 {
      %v5 = affine.load %in_matrix[%i4, %i5] : memref<10x10xf32>
      %v6 = affine.load %in_vec1[%i5] : memref<10xf32>
      %v7 = arith.mulf %v5, %v6 : f32
      %v8 = affine.load %out_vec1[%i5] : memref<10xf32>
      %v9 = arith.addf %v7, %v8 : f32
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
// CHECK-NEXT:       arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:       arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}, 0] : memref<10x1xf32>
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:       arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:       arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
  return
}

// -----

func.func @should_not_slice_past_slice_barrier() {
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

#map = affine_map<(d0, d1) -> (d0 * 16 + d1)>
func.func @fuse_across_dim_mismatch(%arg0: memref<4x4x16x1xf32>, %arg1: memref<144x9xf32>, %arg2: memref<9xf32>) {
  %1 = memref.alloc() : memref<144x4xf32>
  %2 = arith.constant 0.0 : f32
  affine.for %i2 = 0 to 9 {
    affine.for %i3 = 0 to 4 {
      affine.for %i5 = 0 to 16 {
        %7 = affine.apply #map(%i2, %i5)
        affine.store %2, %1[%7, %i3] : memref<144x4xf32>
      }
    }
  }
  affine.for %i6 = 0 to 9 {
    affine.for %i7 = 0 to 9 {
      affine.for %i8 = 0 to 4 {
        affine.for %i10 = 0 to 16 {
          %10 = affine.apply #map(%i6, %i10)
          %11 = affine.load %1[%10, %i8] : memref<144x4xf32>
        }
      }
    }
  }
  return
}
// MAXIMAL:      #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0 * 16 + d1)>
// MAXIMAL-LABEL: func @fuse_across_dim_mismatch
// MAXIMAL:        memref.alloc() : memref<1x1xf32>
// MAXIMAL:        affine.for %{{.*}} = 0 to 9 {
// MAXIMAL-NEXT:    affine.for %{{.*}} = 0 to 9 {
// MAXIMAL-NEXT:      affine.for %{{.*}} = 0 to 4 {
// MAXIMAL-NEXT:        affine.for %{{.*}} = 0 to 16 {
// MAXIMAL-NEXT:          affine.apply #[[$MAP]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:          affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
// MAXIMAL-NEXT:          affine.apply #[[$MAP]](%{{.*}}, %{{.*}})
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
func.func @fuse_across_varying_dims_complex(%arg0: f32) {
  %c0 = arith.constant 0 : index
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
// MAXIMAL-DAG: [[$MAP0:#map[0-9]*]] = affine_map<(d0, d1) -> ((d0 * 72 + d1) floordiv 2304)>
// MAXIMAL-DAG: [[$MAP1:#map[0-9]*]] = affine_map<(d0, d1) -> (((d0 * 72 + d1) mod 2304) floordiv 1152)>
// MAXIMAL-DAG: [[$MAP2:#map[0-9]*]] = affine_map<(d0, d1) -> ((((d0 * 72 + d1) mod 1152) floordiv 9) floordiv 8)>
// MAXIMAL-DAG: [[$MAP3:#map[0-9]*]] = affine_map<(d0, d1) -> ((d1 mod 9) floordiv 3)>
// MAXIMAL-DAG: [[$MAP4:#map[0-9]*]] = affine_map<(d0, d1) -> (d1 mod 3)>
// MAXIMAL-DAG: [[$MAP7:#map[0-9]*]] = affine_map<(d0, d1) -> (d0 * 16 + d1)>
// MAXIMAL-DAG: [[$MAP8:#map[0-9]*]] = affine_map<(d0, d1) -> (d0 * 16 - d1 + 15)>
// MAXIMAL-LABEL: func @fuse_across_varying_dims_complex
// MAXIMAL-NEXT:  memref.alloc() : memref<64x1xf32>
// MAXIMAL-NEXT:  arith.constant 0 : index
// MAXIMAL-NEXT:  memref.alloc() : memref<2x2x3x3x16x1xf32>
// MAXIMAL-NEXT:  memref.alloc() : memref<144x4xf32>
// MAXIMAL-NEXT:  affine.for %{{.*}} = 0 to 9 {
// MAXIMAL-NEXT:    affine.for %{{.*}} = 0 to 4 {
// MAXIMAL-NEXT:      affine.for %{{.*}} = 0 to 16 {
// MAXIMAL-NEXT:        affine.for %{{.*}} = 0 to 64 {
// MAXIMAL-NEXT:          affine.apply [[$MAP0]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:          affine.apply [[$MAP1]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:          affine.apply [[$MAP2]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:          affine.apply [[$MAP3]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:          affine.apply [[$MAP4]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:          affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<2x2x3x3x16x1xf32>
// MAXIMAL-NEXT:          affine.store %{{.*}}, %{{.*}}[%{{.*}}, 0] : memref<64x1xf32>
// MAXIMAL-NEXT:        }
// MAXIMAL-NEXT:        affine.apply [[$MAP7]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:        affine.load %{{.*}}[%{{.*}} * 16 + %{{.*}}, 0] : memref<64x1xf32>
// MAXIMAL-NEXT:        affine.for %{{.*}} = 0 to 9 {
// MAXIMAL-NEXT:          affine.for %{{.*}} = 0 to 4 {
// MAXIMAL-NEXT:            affine.for %{{.*}} = 0 to 16 {
// MAXIMAL-NEXT:              affine.apply [[$MAP8]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:              affine.load %{{.*}}[%{{.*}} * 16 - %{{.*}} + 15, 0] : memref<64x1xf32>
// MAXIMAL-NEXT:            }
// MAXIMAL-NEXT:          }
// MAXIMAL-NEXT:        }
// MAXIMAL-NEXT:      }
// MAXIMAL-NEXT:      affine.for %{{.*}} = 0 to 16 {
// MAXIMAL-NEXT:        affine.apply [[$MAP7]](%{{.*}}, %{{.*}})
// MAXIMAL-NEXT:        affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<144x4xf32>
// MAXIMAL-NEXT:      }
// MAXIMAL-NEXT:    }
// MAXIMAL-NEXT:  }

// -----

func.func @should_fuse_with_slice_union() {
  %a = memref.alloc() : memref<100xf32>
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32

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

func.func @affine_add_mm_fused(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>, %arg3: memref<1024x1024xf32>) {
  affine.for %i2 = 0 to 1024 {
    affine.for %i3 = 0 to 1024 {
      %0 = affine.load %arg3[%i2, %i3] : memref<1024x1024xf32>
      %1 = affine.load %arg2[%i2, %i3] : memref<1024x1024xf32>
      %2 = arith.addf %1, %0 : f32
      affine.store %2, %arg2[%i2, %i3] : memref<1024x1024xf32>
    }
  }
  affine.for %i4 = 0 to 1024 {
    affine.for %i5 = 0 to 1024 {
      affine.for %i6 = 0 to 1024 {
        %3 = affine.load %arg1[%i6, %i5] : memref<1024x1024xf32>
        %4 = affine.load %arg0[%i4, %i6] : memref<1024x1024xf32>
        %5 = arith.mulf %4, %3 : f32
        %6 = affine.load %arg2[%i4, %i5] : memref<1024x1024xf32>
        %7 = arith.addf %6, %5 : f32
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
  // CHECK-NEXT:      arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:      affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:        affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:        affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:        arith.mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:        affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:        arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:        affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  return
}

// -----

func.func @affine_2mm_fused(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>, %arg3: memref<1024x1024xf32>, %arg4: memref<1024x1024xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
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
        %2 = arith.mulf %1, %0 : f32
        %3 = affine.load %arg2[%i4, %i5] : memref<1024x1024xf32>
        %4 = arith.addf %3, %2 : f32
        affine.store %4, %arg2[%i4, %i5] : memref<1024x1024xf32>
      }
    }
  }
  affine.for %i7 = 0 to 1024 {
    affine.for %i8 = 0 to 1024 {
      affine.for %i9 = 0 to 1024 {
        %5 = affine.load %arg1[%i9, %i8] : memref<1024x1024xf32>
        %6 = affine.load %arg0[%i7, %i9] : memref<1024x1024xf32>
        %7 = arith.mulf %6, %5 : f32
        %8 = affine.load %arg4[%i7, %i8] : memref<1024x1024xf32>
        %9 = arith.addf %8, %7 : f32
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
  // CHECK-NEXT:         arith.mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         arith.mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }

  return
}

// -----

func.func @affine_2_dependent_mm_fused(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>, %arg3: memref<1024x1024xf32>, %arg4: memref<1024x1024xf32>) {
  affine.for %i0 = 0 to 1024 {
    affine.for %i1 = 0 to 1024 {
      affine.for %i2 = 0 to 1024 {
        %0 = affine.load %arg1[%i2, %i1] : memref<1024x1024xf32>
        %1 = affine.load %arg0[%i0, %i2] : memref<1024x1024xf32>
        %2 = arith.mulf %1, %0 : f32
        %3 = affine.load %arg2[%i0, %i1] : memref<1024x1024xf32>
        %4 = arith.addf %3, %2 : f32
        affine.store %4, %arg2[%i0, %i1] : memref<1024x1024xf32>
      }
    }
  }
  affine.for %i3 = 0 to 1024 {
    affine.for %i4 = 0 to 1024 {
      affine.for %i5 = 0 to 1024 {
        %5 = affine.load %arg3[%i5, %i4] : memref<1024x1024xf32>
        %6 = affine.load %arg2[%i3, %i5] : memref<1024x1024xf32>
        %7 = arith.mulf %6, %5 : f32
        %8 = affine.load %arg4[%i3, %i4] : memref<1024x1024xf32>
        %9 = arith.addf %8, %7 : f32
        affine.store %9, %arg4[%i3, %i4] : memref<1024x1024xf32>
      }
    }
  }

  // CHECK:        affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:     affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:       affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         arith.mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:       affine.for %{{.*}} = 0 to 1024 {
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         arith.mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:         arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:         affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  return
}

// -----

// CHECK-LABEL: func @should_fuse_self_dependence_multi_store_producer() {
func.func @should_fuse_self_dependence_multi_store_producer() {
  %m = memref.alloc() : memref<10xf32>
  %local_m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

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
func.func @should_fuse_dead_multi_store_producer() {
  %m = memref.alloc() : memref<10xf32>
  %dead_m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

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
func.func @should_fuse_function_live_out_multi_store_producer(%live_in_out_m : memref<10xf32>) {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

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

// Add further tests in mlir/test/Transforms/loop-fusion-4.mlir

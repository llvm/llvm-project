// RUN: mlir-opt -allow-unregistered-dialect %s -affine-scalrep | FileCheck %s

// CHECK-DAG: [[$MAP0:#map[0-9]+]] = affine_map<(d0, d1) -> (d1 + 1)>
// CHECK-DAG: [[$MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: [[$MAP2:#map[0-9]+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: [[$MAP3:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 - 1)>
// CHECK-DAG: [[$MAP4:#map[0-9]+]] = affine_map<(d0) -> (d0 + 1)>

// CHECK-LABEL: func @simple_store_load() {
func.func @simple_store_load() {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    %v0 = affine.load %m[%i0] : memref<10xf32>
    %v1 = arith.addf %v0, %v0 : f32
  }
  return
// CHECK:       %{{.*}} = arith.constant 7.000000e+00 : f32
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  return
}

// CHECK-LABEL: func @multi_store_load() {
func.func @multi_store_load() {
  %c0 = arith.constant 0 : index
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  %cf9 = arith.constant 9.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    %v0 = affine.load %m[%i0] : memref<10xf32>
    %v1 = arith.addf %v0, %v0 : f32
    affine.store %cf8, %m[%i0] : memref<10xf32>
    affine.store %cf9, %m[%i0] : memref<10xf32>
    %v2 = affine.load %m[%i0] : memref<10xf32>
    %v3 = affine.load %m[%i0] : memref<10xf32>
    %v4 = arith.mulf %v2, %v3 : f32
  }
  return
// CHECK:       %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:  %{{.*}} = arith.constant 7.000000e+00 : f32
// CHECK-NEXT:  %{{.*}} = arith.constant 8.000000e+00 : f32
// CHECK-NEXT:  %{{.*}} = arith.constant 9.000000e+00 : f32
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  return

}

// The store-load forwarding can see through affine apply's since it relies on
// dependence information.
// CHECK-LABEL: func @store_load_affine_apply
func.func @store_load_affine_apply() -> memref<10x10xf32> {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %t0 = affine.apply affine_map<(d0, d1) -> (d1 + 1)>(%i0, %i1)
      %t1 = affine.apply affine_map<(d0, d1) -> (d0)>(%i0, %i1)
      %idx0 = affine.apply affine_map<(d0, d1) -> (d1)> (%t0, %t1)
      %idx1 = affine.apply affine_map<(d0, d1) -> (d0 - 1)> (%t0, %t1)
      affine.store %cf7, %m[%idx0, %idx1] : memref<10x10xf32>
      // CHECK-NOT: affine.load %{{[0-9]+}}
      %v0 = affine.load %m[%i0, %i1] : memref<10x10xf32>
      %v1 = arith.addf %v0, %v0 : f32
    }
  }
  // The memref and its stores won't be erased due to this memref return.
  return %m : memref<10x10xf32>
// CHECK:       %{{.*}} = arith.constant 7.000000e+00 : f32
// CHECK-NEXT:  %{{.*}} = memref.alloc() : memref<10x10xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:      %{{.*}} = affine.apply [[$MAP0]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.apply [[$MAP1]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.apply [[$MAP2]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.apply [[$MAP3]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return %{{.*}} : memref<10x10xf32>
}

// CHECK-LABEL: func @store_load_nested
func.func @store_load_nested(%N : index) {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = arith.addf %v0, %v0 : f32
    }
  }
  return
// CHECK:       %{{.*}} = arith.constant 7.000000e+00 : f32
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
}

// No forwarding happens here since either of the two stores could be the last
// writer; store/load forwarding will however be possible here once loop live
// out SSA scalars are available.
// CHECK-LABEL: func @multi_store_load_nested_no_fwd
func.func @multi_store_load_nested_no_fwd(%N : index) {
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      affine.store %cf8, %m[%i1] : memref<10xf32>
    }
    affine.for %i2 = 0 to %N {
      // CHECK: %{{[0-9]+}} = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = arith.addf %v0, %v0 : f32
    }
  }
  return
}

// No forwarding happens here since both stores have a value going into
// the load.
// CHECK-LABEL: func @store_load_store_nested_no_fwd
func.func @store_load_store_nested_no_fwd(%N : index) {
  %cf7 = arith.constant 7.0 : f32
  %cf9 = arith.constant 9.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      // CHECK: %{{[0-9]+}} = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = arith.addf %v0, %v0 : f32
      affine.store %cf9, %m[%i0] : memref<10xf32>
    }
  }
  return
}

// Forwarding happens here since the last store postdominates all other stores
// and other forwarding criteria are satisfied.
// CHECK-LABEL: func @multi_store_load_nested_fwd
func.func @multi_store_load_nested_fwd(%N : index) {
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  %cf9 = arith.constant 9.0 : f32
  %cf10 = arith.constant 10.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      affine.store %cf8, %m[%i1] : memref<10xf32>
    }
    affine.for %i2 = 0 to %N {
      affine.store %cf9, %m[%i2] : memref<10xf32>
    }
    affine.store %cf10, %m[%i0] : memref<10xf32>
    affine.for %i3 = 0 to %N {
      // CHECK-NOT: %{{[0-9]+}} = affine.load
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = arith.addf %v0, %v0 : f32
    }
  }
  return
}

// There is no unique load location for the store to forward to.
// CHECK-LABEL: func @store_load_no_fwd
func.func @store_load_no_fwd() {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        // CHECK: affine.load %{{[0-9]+}}
        %v0 = affine.load %m[%i2] : memref<10xf32>
        %v1 = arith.addf %v0, %v0 : f32
      }
    }
  }
  return
}

// Forwarding happens here as there is a one-to-one store-load correspondence.
// CHECK-LABEL: func @store_load_fwd
func.func @store_load_fwd() {
  %cf7 = arith.constant 7.0 : f32
  %c0 = arith.constant 0 : index
  %m = memref.alloc() : memref<10xf32>
  affine.store %cf7, %m[%c0] : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        // CHECK-NOT: affine.load %{{[0-9]}}+
        %v0 = affine.load %m[%c0] : memref<10xf32>
        %v1 = arith.addf %v0, %v0 : f32
      }
    }
  }
  return
}

// Although there is a dependence from the second store to the load, it is
// satisfied by the outer surrounding loop, and does not prevent the first
// store to be forwarded to the load.
func.func @store_load_store_nested_fwd(%N : index) -> f32 {
  %cf7 = arith.constant 7.0 : f32
  %cf9 = arith.constant 9.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = arith.addf %v0, %v0 : f32
      %idx = affine.apply affine_map<(d0) -> (d0 + 1)> (%i0)
      affine.store %cf9, %m[%idx] : memref<10xf32>
    }
  }
  // Due to this load, the memref isn't optimized away.
  %v3 = affine.load %m[%c1] : memref<10xf32>
  return %v3 : f32
// CHECK:       %{{.*}} = memref.alloc() : memref<10xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:    affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:      %{{.*}} = affine.apply [[$MAP4]](%{{.*}})
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:  return %{{.*}} : f32
}

// CHECK-LABEL: func @should_not_fwd
func.func @should_not_fwd(%A: memref<100xf32>, %M : index, %N : index) -> f32 {
  %cf = arith.constant 0.0 : f32
  affine.store %cf, %A[%M] : memref<100xf32>
  // CHECK: affine.load %{{.*}}[%{{.*}}]
  %v = affine.load %A[%N] : memref<100xf32>
  return %v : f32
}

// Can store forward to A[%j, %i], but no forwarding to load on %A[%i, %j]
// CHECK-LABEL: func @refs_not_known_to_be_equal
func.func @refs_not_known_to_be_equal(%A : memref<100 x 100 x f32>, %M : index) {
  %N = affine.apply affine_map<(d0) -> (d0 + 1)> (%M)
  %cf1 = arith.constant 1.0 : f32
  affine.for %i = 0 to 100 {
  // CHECK: affine.for %[[I:.*]] =
    affine.for %j = 0 to 100 {
    // CHECK: affine.for %[[J:.*]] =
      // CHECK: affine.load %{{.*}}[%[[I]], %[[J]]]
      %u = affine.load %A[%i, %j] : memref<100x100xf32>
      // CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[%[[J]], %[[I]]]
      affine.store %cf1, %A[%j, %i] : memref<100x100xf32>
      // CHECK-NEXT: affine.load %{{.*}}[%[[I]], %[[J]]]
      %v = affine.load %A[%i, %j] : memref<100x100xf32>
      // This load should disappear.
      %w = affine.load %A[%j, %i] : memref<100x100xf32>
      // CHECK-NEXT: "foo"
      "foo" (%u, %v, %w) : (f32, f32, f32) -> ()
    }
  }
  return
}

// The test checks for value forwarding from vector stores to vector loads.
// The value loaded from %in can directly be stored to %out by eliminating
// store and load from %tmp.
func.func @vector_forwarding(%in : memref<512xf32>, %out : memref<512xf32>) {
  %tmp = memref.alloc() : memref<512xf32>
  affine.for %i = 0 to 16 {
    %ld0 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    affine.vector_store %ld0, %tmp[32*%i] : memref<512xf32>, vector<32xf32>
    %ld1 = affine.vector_load %tmp[32*%i] : memref<512xf32>, vector<32xf32>
    affine.vector_store %ld1, %out[32*%i] : memref<512xf32>, vector<32xf32>
  }
  return
}

// CHECK-LABEL: func @vector_forwarding
// CHECK:      affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:   %[[LDVAL:.*]] = affine.vector_load
// CHECK-NEXT:   affine.vector_store %[[LDVAL]],{{.*}}
// CHECK-NEXT: }

func.func @vector_no_forwarding(%in : memref<512xf32>, %out : memref<512xf32>) {
  %tmp = memref.alloc() : memref<512xf32>
  affine.for %i = 0 to 16 {
    %ld0 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    affine.vector_store %ld0, %tmp[32*%i] : memref<512xf32>, vector<32xf32>
    %ld1 = affine.vector_load %tmp[32*%i] : memref<512xf32>, vector<16xf32>
    affine.vector_store %ld1, %out[32*%i] : memref<512xf32>, vector<16xf32>
  }
  return
}

// CHECK-LABEL: func @vector_no_forwarding
// CHECK:      affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:   %[[LDVAL:.*]] = affine.vector_load
// CHECK-NEXT:   affine.vector_store %[[LDVAL]],{{.*}}
// CHECK-NEXT:   %[[LDVAL1:.*]] = affine.vector_load
// CHECK-NEXT:   affine.vector_store %[[LDVAL1]],{{.*}}
// CHECK-NEXT: }

// CHECK-LABEL: func @simple_three_loads
func.func @simple_three_loads(%in : memref<10xf32>) {
  affine.for %i0 = 0 to 10 {
    // CHECK:       affine.load
    %v0 = affine.load %in[%i0] : memref<10xf32>
    // CHECK-NOT:   affine.load
    %v1 = affine.load %in[%i0] : memref<10xf32>
    %v2 = arith.addf %v0, %v1 : f32
    %v3 = affine.load %in[%i0] : memref<10xf32>
    %v4 = arith.addf %v2, %v3 : f32
  }
  return
}

// CHECK-LABEL: func @nested_loads_const_index
func.func @nested_loads_const_index(%in : memref<10xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK:       affine.load
  %v0 = affine.load %in[%c0] : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 20 {
      affine.for %i2 = 0 to 30 {
        // CHECK-NOT:   affine.load
        %v1 = affine.load %in[%c0] : memref<10xf32>
        %v2 = arith.addf %v0, %v1 : f32
      }
    }
  }
  return
}

// CHECK-LABEL: func @nested_loads
func.func @nested_loads(%N : index, %in : memref<10xf32>) {
  affine.for %i0 = 0 to 10 {
    // CHECK:       affine.load
    %v0 = affine.load %in[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      // CHECK-NOT:   affine.load
      %v1 = affine.load %in[%i0] : memref<10xf32>
      %v2 = arith.addf %v0, %v1 : f32
    }
  }
  return
}

// CHECK-LABEL: func @nested_loads_different_memref_accesses_no_cse
func.func @nested_loads_different_memref_accesses_no_cse(%in : memref<10xf32>) {
  affine.for %i0 = 0 to 10 {
    // CHECK:       affine.load
    %v0 = affine.load %in[%i0] : memref<10xf32>
    affine.for %i1 = 0 to 20 {
      // CHECK:       affine.load
      %v1 = affine.load %in[%i1] : memref<10xf32>
      %v2 = arith.addf %v0, %v1 : f32
    }
  }
  return
}

// CHECK-LABEL: func @load_load_store
func.func @load_load_store(%m : memref<10xf32>) {
  affine.for %i0 = 0 to 10 {
    // CHECK:       affine.load
    %v0 = affine.load %m[%i0] : memref<10xf32>
    // CHECK-NOT:       affine.load
    %v1 = affine.load %m[%i0] : memref<10xf32>
    %v2 = arith.addf %v0, %v1 : f32
    affine.store %v2, %m[%i0] : memref<10xf32>
  }
  return
}

// CHECK-LABEL: func @load_load_store_2_loops_no_cse
func.func @load_load_store_2_loops_no_cse(%N : index, %m : memref<10xf32>) {
  affine.for %i0 = 0 to 10 {
    // CHECK:       affine.load
    %v0 = affine.load %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      // CHECK:       affine.load
      %v1 = affine.load %m[%i0] : memref<10xf32>
      %v2 = arith.addf %v0, %v1 : f32
      affine.store %v2, %m[%i0] : memref<10xf32>
    }
  }
  return
}

// CHECK-LABEL: func @load_load_store_3_loops_no_cse
func.func @load_load_store_3_loops_no_cse(%m : memref<10xf32>) {
%cf1 = arith.constant 1.0 : f32
  affine.for %i0 = 0 to 10 {
    // CHECK:       affine.load
    %v0 = affine.load %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to 20 {
      affine.for %i2 = 0 to 30 {
        // CHECK:       affine.load
        %v1 = affine.load %m[%i0] : memref<10xf32>
        %v2 = arith.addf %v0, %v1 : f32
      }
      affine.store %cf1, %m[%i0] : memref<10xf32>
    }
  }
  return
}

// CHECK-LABEL: func @load_load_store_3_loops
func.func @load_load_store_3_loops(%m : memref<10xf32>) {
%cf1 = arith.constant 1.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 20 {
      // CHECK:       affine.load
      %v0 = affine.load %m[%i0] : memref<10xf32>
      affine.for %i2 = 0 to 30 {
        // CHECK-NOT:   affine.load
        %v1 = affine.load %m[%i0] : memref<10xf32>
        %v2 = arith.addf %v0, %v1 : f32
      }
    }
    affine.store %cf1, %m[%i0] : memref<10xf32>
  }
  return
}

// CHECK-LABEL: func @loads_in_sibling_loops_const_index_no_cse
func.func @loads_in_sibling_loops_const_index_no_cse(%m : memref<10xf32>) {
  %c0 = arith.constant 0 : index
  affine.for %i0 = 0 to 10 {
    // CHECK:       affine.load
    %v0 = affine.load %m[%c0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    // CHECK:       affine.load
    %v0 = affine.load %m[%c0] : memref<10xf32>
    %v1 = arith.addf %v0, %v0 : f32
  }
  return
}

// CHECK-LABEL: func @load_load_affine_apply
func.func @load_load_affine_apply(%in : memref<10x10xf32>) {
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %t0 = affine.apply affine_map<(d0, d1) -> (d1 + 1)>(%i0, %i1)
      %t1 = affine.apply affine_map<(d0, d1) -> (d0)>(%i0, %i1)
      %idx0 = affine.apply affine_map<(d0, d1) -> (d1)> (%t0, %t1)
      %idx1 = affine.apply affine_map<(d0, d1) -> (d0 - 1)> (%t0, %t1)
      // CHECK:       affine.load
      %v0 = affine.load %in[%idx0, %idx1] : memref<10x10xf32>
      // CHECK-NOT:   affine.load
      %v1 = affine.load %in[%i0, %i1] : memref<10x10xf32>
      %v2 = arith.addf %v0, %v1 : f32
    }
  }
  return
}

// CHECK-LABEL: func @vector_loads
func.func @vector_loads(%in : memref<512xf32>, %out : memref<512xf32>) {
  affine.for %i = 0 to 16 {
    // CHECK:       affine.vector_load
    %ld0 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    // CHECK-NOT:   affine.vector_load
    %ld1 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    %add = arith.addf %ld0, %ld1 : vector<32xf32>
    affine.vector_store %ld1, %out[32*%i] : memref<512xf32>, vector<32xf32>
  }
  return
}

// CHECK-LABEL: func @vector_loads_no_cse
func.func @vector_loads_no_cse(%in : memref<512xf32>, %out : memref<512xf32>) {
  affine.for %i = 0 to 16 {
    // CHECK:       affine.vector_load
    %ld0 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    // CHECK:   affine.vector_load
    %ld1 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<16xf32>
    affine.vector_store %ld1, %out[32*%i] : memref<512xf32>, vector<16xf32>
  }
  return
}

// CHECK-LABEL: func @vector_load_store_load_no_cse
func.func @vector_load_store_load_no_cse(%in : memref<512xf32>, %out : memref<512xf32>) {
  affine.for %i = 0 to 16 {
    // CHECK:       affine.vector_load
    %ld0 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    affine.vector_store %ld0, %in[16*%i] : memref<512xf32>, vector<32xf32>
    // CHECK:       affine.vector_load
    %ld1 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    %add = arith.addf %ld0, %ld1 : vector<32xf32>
    affine.vector_store %ld1, %out[32*%i] : memref<512xf32>, vector<32xf32>
  }
  return
}

// CHECK-LABEL: func @reduction_multi_store
func.func @reduction_multi_store() -> memref<1xf32> {
  %A = memref.alloc() : memref<1xf32>
  %cf0 = arith.constant 0.0 : f32
  %cf5 = arith.constant 5.0 : f32

 affine.store %cf0, %A[0] : memref<1xf32>
  affine.for %i = 0 to 100 step 2 {
    %l = affine.load %A[0] : memref<1xf32>
    %s = arith.addf %l, %cf5 : f32
    // Store to load forwarding from this store should happen.
    affine.store %s, %A[0] : memref<1xf32>
    %m = affine.load %A[0] : memref<1xf32>
   "test.foo"(%m) : (f32) -> ()
  }

// CHECK:       affine.for
// CHECK:         affine.load
// CHECK:         affine.store %[[S:.*]],
// CHECK-NEXT:    "test.foo"(%[[S]])

  return %A : memref<1xf32>
}

// CHECK-LABEL: func @vector_load_affine_apply_store_load
func.func @vector_load_affine_apply_store_load(%in : memref<512xf32>, %out : memref<512xf32>) {
  %cf1 = arith.constant 1: index
  affine.for %i = 0 to 15 {
    // CHECK:       affine.vector_load
    %ld0 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    %idx = affine.apply affine_map<(d0) -> (d0 + 1)> (%i)
    affine.vector_store %ld0, %in[32*%idx] : memref<512xf32>, vector<32xf32>
    // CHECK-NOT:   affine.vector_load
    %ld1 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    %add = arith.addf %ld0, %ld1 : vector<32xf32>
    affine.vector_store %ld1, %out[32*%i] : memref<512xf32>, vector<32xf32>
  }
  return
}

// CHECK-LABEL: func @external_no_forward_load

func.func @external_no_forward_load(%in : memref<512xf32>, %out : memref<512xf32>) {
  affine.for %i = 0 to 16 {
    %ld0 = affine.load %in[32*%i] : memref<512xf32>
    affine.store %ld0, %out[32*%i] : memref<512xf32>
    "memop"(%in, %out) : (memref<512xf32>, memref<512xf32>) -> ()
    %ld1 = affine.load %in[32*%i] : memref<512xf32>
    affine.store %ld1, %out[32*%i] : memref<512xf32>
  }
  return
}
// CHECK:   affine.load
// CHECK:   affine.store
// CHECK:   affine.load
// CHECK:   affine.store

// CHECK-LABEL: func @external_no_forward_store

func.func @external_no_forward_store(%in : memref<512xf32>, %out : memref<512xf32>) {
  %cf1 = arith.constant 1.0 : f32
  affine.for %i = 0 to 16 {
    affine.store %cf1, %in[32*%i] : memref<512xf32>
    "memop"(%in, %out) : (memref<512xf32>, memref<512xf32>) -> ()
    %ld1 = affine.load %in[32*%i] : memref<512xf32>
    affine.store %ld1, %out[32*%i] : memref<512xf32>
  }
  return
}
// CHECK:   affine.store
// CHECK:   affine.load
// CHECK:   affine.store

// CHECK-LABEL: func @no_forward_cast

func.func @no_forward_cast(%in : memref<512xf32>, %out : memref<512xf32>) {
  %cf1 = arith.constant 1.0 : f32
  %cf2 = arith.constant 2.0 : f32
  %m2 = memref.cast %in : memref<512xf32> to memref<?xf32>
  affine.for %i = 0 to 16 {
    affine.store %cf1, %in[32*%i] : memref<512xf32>
    affine.store %cf2, %m2[32*%i] : memref<?xf32>
    %ld1 = affine.load %in[32*%i] : memref<512xf32>
    affine.store %ld1, %out[32*%i] : memref<512xf32>
  }
  return
}
// CHECK:   affine.store
// CHECK-NEXT:   affine.store
// CHECK-NEXT:   affine.load
// CHECK-NEXT:   affine.store

// Although there is a dependence from the second store to the load, it is
// satisfied by the outer surrounding loop, and does not prevent the first
// store to be forwarded to the load.

// CHECK-LABEL: func @overlap_no_fwd
func.func @overlap_no_fwd(%N : index) -> f32 {
  %cf7 = arith.constant 7.0 : f32
  %cf9 = arith.constant 9.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 5 {
    affine.store %cf7, %m[2 * %i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      %v0 = affine.load %m[2 * %i0] : memref<10xf32>
      %v1 = arith.addf %v0, %v0 : f32
      affine.store %cf9, %m[%i0 + 1] : memref<10xf32>
    }
  }
  // Due to this load, the memref isn't optimized away.
  %v3 = affine.load %m[%c1] : memref<10xf32>
  return %v3 : f32

// CHECK:  affine.for %{{.*}} = 0 to 5 {
// CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:    affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK-NEXT:      %{{.*}} = affine.load
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:  return %{{.*}} : f32
}

// CHECK-LABEL: func @redundant_store_elim

func.func @redundant_store_elim(%out : memref<512xf32>) {
  %cf1 = arith.constant 1.0 : f32
  %cf2 = arith.constant 2.0 : f32
  affine.for %i = 0 to 16 {
    affine.store %cf1, %out[32*%i] : memref<512xf32>
    affine.store %cf2, %out[32*%i] : memref<512xf32>
  }
  return
}

// CHECK: affine.for
// CHECK-NEXT:   affine.store
// CHECK-NEXT: }

// CHECK-LABEL: func @redundant_store_elim_fail

func.func @redundant_store_elim_fail(%out : memref<512xf32>) {
  %cf1 = arith.constant 1.0 : f32
  %cf2 = arith.constant 2.0 : f32
  affine.for %i = 0 to 16 {
    affine.store %cf1, %out[32*%i] : memref<512xf32>
    "test.use"(%out) : (memref<512xf32>) -> ()
    affine.store %cf2, %out[32*%i] : memref<512xf32>
  }
  return
}
// CHECK: affine.for
// CHECK-NEXT:   affine.store
// CHECK-NEXT:   "test.use"
// CHECK-NEXT:   affine.store
// CHECK-NEXT: }

// CHECK-LABEL: @with_inner_ops
func.func @with_inner_ops(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i1) {
  %cst = arith.constant 0.000000e+00 : f64
  %cst_0 = arith.constant 3.140000e+00 : f64
  %cst_1 = arith.constant 1.000000e+00 : f64
  affine.for %arg3 = 0 to 28 {
    affine.store %cst, %arg1[%arg3] : memref<?xf64>
    affine.store %cst_0, %arg1[%arg3] : memref<?xf64>
    %0 = scf.if %arg2 -> (f64) {
      scf.yield %cst_1 : f64
    } else {
      %1 = affine.load %arg1[%arg3] : memref<?xf64>
      scf.yield %1 : f64
    }
    affine.store %0, %arg0[%arg3] : memref<?xf64>
  }
  return
}

// CHECK:  %[[pi:.+]] = arith.constant 3.140000e+00 : f64
// CHECK:  %{{.*}} = scf.if %arg2 -> (f64) {
// CHECK:        scf.yield %{{.*}} : f64
// CHECK:      } else {
// CHECK:        scf.yield %[[pi]] : f64
// CHECK:      }

// Check if scalar replacement works correctly when affine memory ops are in the
// body of an scf.for.

// CHECK-LABEL: func @affine_store_load_in_scope
func.func @affine_store_load_in_scope(%memref: memref<1x4094x510x1xf32>, %memref_2: memref<4x4x1x64xf32>, %memref_0: memref<1x2046x254x1x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %c768 = arith.constant 768 : index
  scf.for %i = %c0 to %c768 step %c1 {
    %9 = arith.remsi %i, %c64 : index
    %10 = arith.divsi %i, %c64 : index
    %11 = arith.remsi %10, %c2 : index
    %12 = arith.divsi %10, %c2 : index
    test.affine_scope {
      %14 = arith.muli %12, %c2 : index
      %15 = arith.addi %c2, %14 : index
      %16 = arith.addi %15, %c0 : index
      %18 = arith.muli %11, %c2 : index
      %19 = arith.addi %c2, %18 : index
      %20 = affine.load %memref[0, symbol(%16), symbol(%19), 0] : memref<1x4094x510x1xf32>
      %21 = affine.load %memref_2[0, 0, 0, symbol(%9)] : memref<4x4x1x64xf32>
      %24 = affine.load %memref_0[0, symbol(%12), symbol(%11), 0, symbol(%9)] : memref<1x2046x254x1x64xf32>
      %25 = arith.mulf %20, %21 : f32
      %26 = arith.addf %24, %25 : f32
      // CHECK: %[[A:.*]] = arith.addf
      affine.store %26, %memref_0[0, symbol(%12), symbol(%11), 0, symbol(%9)] : memref<1x2046x254x1x64xf32>
      %27 = arith.addi %19, %c1 : index
      %28 = affine.load %memref[0, symbol(%16), symbol(%27), 0] : memref<1x4094x510x1xf32>
      %29 = affine.load %memref_2[0, 1, 0, symbol(%9)] : memref<4x4x1x64xf32>
      %30 = affine.load %memref_0[0, symbol(%12), symbol(%11), 0, symbol(%9)] : memref<1x2046x254x1x64xf32>
      %31 = arith.mulf %28, %29 : f32
      %32 = arith.addf %30, %31 : f32
      // The addf above will get the forwarded value from the store on
      // %memref_0 above which is being loaded into %30..
      // CHECK: arith.addf %[[A]],
      "terminate"() : () -> ()
    }
  }
  return
}

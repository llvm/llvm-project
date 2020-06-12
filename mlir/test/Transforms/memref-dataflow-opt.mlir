// RUN: mlir-opt -allow-unregistered-dialect %s -memref-dataflow-opt | FileCheck %s

// CHECK-DAG: [[$MAP0:#map[0-9]+]] = affine_map<(d0, d1) -> (d1 + 1)>
// CHECK-DAG: [[$MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: [[$MAP2:#map[0-9]+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: [[$MAP3:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 - 1)>
// CHECK-DAG: [[$MAP4:#map[0-9]+]] = affine_map<(d0) -> (d0 + 1)>

// CHECK-LABEL: func @simple_store_load() {
func @simple_store_load() {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    %v0 = affine.load %m[%i0] : memref<10xf32>
    %v1 = addf %v0, %v0 : f32
  }
  return
// CHECK:       %{{.*}} = constant 7.000000e+00 : f32
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  return
}

// CHECK-LABEL: func @multi_store_load() {
func @multi_store_load() {
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32
  %cf9 = constant 9.0 : f32
  %m = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    %v0 = affine.load %m[%i0] : memref<10xf32>
    %v1 = addf %v0, %v0 : f32
    affine.store %cf8, %m[%i0] : memref<10xf32>
    affine.store %cf9, %m[%i0] : memref<10xf32>
    %v2 = affine.load %m[%i0] : memref<10xf32>
    %v3 = affine.load %m[%i0] : memref<10xf32>
    %v4 = mulf %v2, %v3 : f32
  }
  return
// CHECK:       %{{.*}} = constant 0 : index
// CHECK-NEXT:  %{{.*}} = constant 7.000000e+00 : f32
// CHECK-NEXT:  %{{.*}} = constant 8.000000e+00 : f32
// CHECK-NEXT:  %{{.*}} = constant 9.000000e+00 : f32
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:    %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  return

}

// The store-load forwarding can see through affine apply's since it relies on
// dependence information.
// CHECK-LABEL: func @store_load_affine_apply
func @store_load_affine_apply() -> memref<10x10xf32> {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %t0 = affine.apply affine_map<(d0, d1) -> (d1 + 1)>(%i0, %i1)
      %t1 = affine.apply affine_map<(d0, d1) -> (d0)>(%i0, %i1)
      %idx0 = affine.apply affine_map<(d0, d1) -> (d1)> (%t0, %t1)
      %idx1 = affine.apply affine_map<(d0, d1) -> (d0 - 1)> (%t0, %t1)
      affine.store %cf7, %m[%idx0, %idx1] : memref<10x10xf32>
      // CHECK-NOT: affine.load %{{[0-9]+}}
      %v0 = affine.load %m[%i0, %i1] : memref<10x10xf32>
      %v1 = addf %v0, %v0 : f32
    }
  }
  // The memref and its stores won't be erased due to this memref return.
  return %m : memref<10x10xf32>
// CHECK:       %{{.*}} = constant 7.000000e+00 : f32
// CHECK-NEXT:  %{{.*}} = alloc() : memref<10x10xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:      %{{.*}} = affine.apply [[$MAP0]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.apply [[$MAP1]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.apply [[$MAP2]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.apply [[$MAP3]](%{{.*}}, %{{.*}})
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:      %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return %{{.*}} : memref<10x10xf32>
}

// CHECK-LABEL: func @store_load_nested
func @store_load_nested(%N : index) {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = addf %v0, %v0 : f32
    }
  }
  return
// CHECK:       %{{.*}} = constant 7.000000e+00 : f32
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK-NEXT:      %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
}

// No forwarding happens here since either of the two stores could be the last
// writer; store/load forwarding will however be possible here once loop live
// out SSA scalars are available.
// CHECK-LABEL: func @multi_store_load_nested_no_fwd
func @multi_store_load_nested_no_fwd(%N : index) {
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32
  %m = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      affine.store %cf8, %m[%i1] : memref<10xf32>
    }
    affine.for %i2 = 0 to %N {
      // CHECK: %{{[0-9]+}} = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = addf %v0, %v0 : f32
    }
  }
  return
}

// No forwarding happens here since both stores have a value going into
// the load.
// CHECK-LABEL: func @store_load_store_nested_no_fwd
func @store_load_store_nested_no_fwd(%N : index) {
  %cf7 = constant 7.0 : f32
  %cf9 = constant 9.0 : f32
  %m = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      // CHECK: %{{[0-9]+}} = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = addf %v0, %v0 : f32
      affine.store %cf9, %m[%i0] : memref<10xf32>
    }
  }
  return
}

// Forwarding happens here since the last store postdominates all other stores
// and other forwarding criteria are satisfied.
// CHECK-LABEL: func @multi_store_load_nested_fwd
func @multi_store_load_nested_fwd(%N : index) {
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32
  %cf9 = constant 9.0 : f32
  %cf10 = constant 10.0 : f32
  %m = alloc() : memref<10xf32>
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
      %v1 = addf %v0, %v0 : f32
    }
  }
  return
}

// There is no unique load location for the store to forward to.
// CHECK-LABEL: func @store_load_no_fwd
func @store_load_no_fwd() {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        // CHECK: affine.load %{{[0-9]+}}
        %v0 = affine.load %m[%i2] : memref<10xf32>
        %v1 = addf %v0, %v0 : f32
      }
    }
  }
  return
}

// Forwarding happens here as there is a one-to-one store-load correspondence.
// CHECK-LABEL: func @store_load_fwd
func @store_load_fwd() {
  %cf7 = constant 7.0 : f32
  %c0 = constant 0 : index
  %m = alloc() : memref<10xf32>
  affine.store %cf7, %m[%c0] : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        // CHECK-NOT: affine.load %{{[0-9]}}+
        %v0 = affine.load %m[%c0] : memref<10xf32>
        %v1 = addf %v0, %v0 : f32
      }
    }
  }
  return
}

// Although there is a dependence from the second store to the load, it is
// satisfied by the outer surrounding loop, and does not prevent the first
// store to be forwarded to the load.
func @store_load_store_nested_fwd(%N : index) -> f32 {
  %cf7 = constant 7.0 : f32
  %cf9 = constant 9.0 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %m = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = addf %v0, %v0 : f32
      %idx = affine.apply affine_map<(d0) -> (d0 + 1)> (%i0)
      affine.store %cf9, %m[%idx] : memref<10xf32>
    }
  }
  // Due to this load, the memref isn't optimized away.
  %v3 = affine.load %m[%c1] : memref<10xf32>
  return %v3 : f32
// CHECK:       %{{.*}} = alloc() : memref<10xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:    affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK-NEXT:      %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:      %{{.*}} = affine.apply [[$MAP4]](%{{.*}})
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:  return %{{.*}} : f32
}

// CHECK-LABEL: func @should_not_fwd
func @should_not_fwd(%A: memref<100xf32>, %M : index, %N : index) -> f32 {
  %cf = constant 0.0 : f32
  affine.store %cf, %A[%M] : memref<100xf32>
  // CHECK: affine.load %{{.*}}[%{{.*}}]
  %v = affine.load %A[%N] : memref<100xf32>
  return %v : f32
}

// Can store forward to A[%j, %i], but no forwarding to load on %A[%i, %j]
// CHECK-LABEL: func @refs_not_known_to_be_equal
func @refs_not_known_to_be_equal(%A : memref<100 x 100 x f32>, %M : index) {
  %N = affine.apply affine_map<(d0) -> (d0 + 1)> (%M)
  %cf1 = constant 1.0 : f32
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

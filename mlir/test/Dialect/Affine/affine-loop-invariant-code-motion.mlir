// RUN: mlir-opt %s -affine-loop-invariant-code-motion -split-input-file | FileCheck %s

func.func @nested_loops_both_having_invariant_code() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = arith.addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      affine.store %v0, %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst_0]] : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.store

  return
}

// -----

// The store-load forwarding can see through affine apply's since it relies on
// dependence information.
// CHECK-LABEL: func @store_affine_apply
func.func @store_affine_apply() -> memref<10xf32> {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %arg0 = 0 to 10 {
      %t0 = affine.apply affine_map<(d1) -> (d1 + 1)>(%arg0)
      affine.store %cf7, %m[%t0] : memref<10xf32>
  }
  return %m : memref<10xf32>
// CHECK:       %[[cst:.*]] = arith.constant 7.000000e+00 : f32
// CHECK-NEXT:  %[[VAR_0:.*]] = memref.alloc() : memref<10xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:      affine.apply
// CHECK-NEXT:      affine.store %[[cst]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[VAR_0]]  : memref<10xf32>
}

// -----

func.func @nested_loops_code_invariant_to_both() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      %v0 = arith.addf %cf7, %cf8 : f32
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst_0]] : f32

  return
}

// -----

// CHECK-LABEL: func @nested_loops_inner_loops_invariant_to_outermost_loop
func.func @nested_loops_inner_loops_invariant_to_outermost_loop(%m : memref<10xindex>) {
  affine.for %arg0 = 0 to 20 {
    affine.for %arg1 = 0 to 30 {
      %v0 = affine.for %arg2 = 0 to 10 iter_args (%prevAccum = %arg1) -> index {
        %v1 = affine.load %m[%arg2] : memref<10xindex>
        %newAccum = arith.addi %prevAccum, %v1 : index
        affine.yield %newAccum : index
      }
    }
  }

  // CHECK:      affine.for %{{.*}} = 0 to 30 {
  // CHECK-NEXT:   %{{.*}}  = affine.for %{{.*}}  = 0 to 10 iter_args(%{{.*}} = %{{.*}}) -> (index) {
  // CHECK-NEXT:     %{{.*}}  = affine.load %{{.*}}[%{{.*}}  : memref<10xindex>
  // CHECK-NEXT:     %{{.*}}  = arith.addi %{{.*}}, %{{.*}} : index
  // CHECK-NEXT:     affine.yield %{{.*}} : index
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 20 {
  // CHECK-NEXT: }

  return
}

// -----

func.func @single_loop_nothing_invariant() {
  %m1 = memref.alloc() : memref<10xf32>
  %m2 = memref.alloc() : memref<11xf32>
  affine.for %arg0 = 0 to 10 {
    %v0 = affine.load %m1[%arg0] : memref<10xf32>
    %v1 = affine.load %m2[%arg0] : memref<11xf32>
    %v2 = arith.addf %v0, %v1 : f32
    affine.store %v2, %m1[%arg0] : memref<10xf32>
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: memref.alloc() : memref<11xf32>
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.load %{{.*}} : memref<10xf32>
  // CHECK-NEXT: affine.load %{{.*}} : memref<11xf32>
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: affine.store %{{.*}} : memref<10xf32>

  return
}

// -----

func.func @invariant_code_inside_affine_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %t0 = affine.apply affine_map<(d1) -> (d1 + 1)>(%arg0)
    affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %t0) {
        %cf9 = arith.addf %cf8, %cf8 : f32
        affine.store %cf9, %m[%arg0] : memref<10xf32>

    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.apply #map{{[0-9]*}}(%arg0)
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: }


  return
}

// -----

func.func @dependent_stores() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = arith.addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = arith.mulf %cf7, %cf7 : f32
      affine.store %v1, %m[%arg1] : memref<10xf32>
      affine.store %v0, %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst_0]] : f32
  // CHECK-NEXT: %[[mul:.*]] = arith.mulf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {

  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %[[mul]]
  // CHECK-NEXT:   affine.store

  return
}

// -----

func.func @independent_stores() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = arith.addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = arith.mulf %cf7, %cf7 : f32
      affine.store %v0, %m[%arg0] : memref<10xf32>
      affine.store %v1, %m[%arg1] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: %[[add:.*]] = arith.addf %[[cst]], %[[cst_0]] : f32
  // CHECK-NEXT: %[[mul:.*]] = arith.mulf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:     affine.store %[[add]]
  // CHECK-NEXT:     affine.store %[[mul]]
  // CHECK-NEXT:    }

  return
}

// -----

func.func @load_dependent_store() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = arith.addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = arith.addf %cf7, %cf7 : f32
      affine.store %v0, %m[%arg1] : memref<10xf32>
      %v2 = affine.load %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst_0]] : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.for
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT:   affine.load

  return
}

// -----

func.func @load_after_load() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = arith.addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = arith.addf %cf7, %cf7 : f32
      %v3 = affine.load %m[%arg1] : memref<10xf32>
      %v2 = affine.load %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst_0]] : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>

  return
}

// -----

func.func @invariant_affine_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>

      }
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_affine_if2() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg1] : memref<10xf32>

      }
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.if
  // CHECK-NEXT:   arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_affine_nested_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>
          affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
            affine.store %cf9, %m[%arg1] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %[[arg0:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.for %[[arg1:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.store {{.*}}[%[[arg0]]] : memref<10xf32>
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: affine.store {{.*}}[%[[arg1]]] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_affine_nested_if_else() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>
          affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
            affine.store %cf9, %m[%arg0] : memref<10xf32>
          } else {
            affine.store %cf9, %m[%arg1] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %[[arg0:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.for %[[arg1:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.store {{.*}}[%[[arg0]]] : memref<10xf32>
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: affine.store {{.*}}[%[[arg0]]] : memref<10xf32>
  // CHECK-NEXT: } else {
  // CHECK-NEXT: affine.store {{.*}}[%[[arg1]]] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_affine_nested_if_else2() {
  %m = memref.alloc() : memref<10xf32>
  %m2 = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
          %tload1 = affine.load %m[%arg0] : memref<10xf32>
          affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
            affine.store %cf9, %m2[%arg0] : memref<10xf32>
          } else {
            %tload2 = affine.load %m[%arg0] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %[[arg0:.*]] = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.load {{.*}}[%[[arg0]]] : memref<10xf32>
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: affine.store {{.*}}[%[[arg0]]] : memref<10xf32>
  // CHECK-NEXT: } else {
  // CHECK-NEXT: affine.load {{.*}}[%[[arg0]]] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_affine_nested_if2() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
          %v1 = affine.load %m[%arg0] : memref<10xf32>
          affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
            %v2 = affine.load %m[%arg0] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %[[arg0:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.load {{.*}}[%[[arg0]]] : memref<10xf32>
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: affine.load {{.*}}[%[[arg0]]] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_affine_for_inside_affine_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>
          affine.for %arg2 = 0 to 10 {
            affine.store %cf9, %m[%arg2] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %[[arg0:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.for %[[arg1:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.store {{.*}}[%[[arg0]]] : memref<10xf32>
  // CHECK-NEXT: affine.for %[[arg2:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.store {{.*}}[%[[arg2]]] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_constant_and_load() {
  %m = memref.alloc() : memref<100xf32>
  %m2 = memref.alloc() : memref<100xf32>
  affine.for %arg0 = 0 to 5 {
    %c0 = arith.constant 0 : index
    %v = affine.load %m2[%c0] : memref<100xf32>
    affine.store %v, %m[%arg0] : memref<100xf32>
  }

  // CHECK: memref.alloc() : memref<100xf32>
  // CHECK-NEXT: memref.alloc() : memref<100xf32>
  // CHECK-NEXT: arith.constant 0 : index
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 5 {
  // CHECK-NEXT:  affine.store


  return
}

// -----

func.func @nested_load_store_same_memref() {
  %m = memref.alloc() : memref<10xf32>
  %cst = arith.constant 8.0 : f32
  %c0 = arith.constant 0 : index
   affine.for %arg0 = 0 to 10 {
    %v0 = affine.load %m[%c0] : memref<10xf32>
    affine.for %arg1 = 0 to 10 {
      affine.store %cst, %m[%arg1] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.constant 0 : index
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.for
  // CHECK-NEXT:    affine.store %[[cst]]


  return
}

// -----

func.func @nested_load_store_same_memref2() {
  %m = memref.alloc() : memref<10xf32>
  %cst = arith.constant 8.0 : f32
  %c0 = arith.constant 0 : index
   affine.for %arg0 = 0 to 10 {
     affine.store %cst, %m[%c0] : memref<10xf32>
      affine.for %arg1 = 0 to 10 {
        %v0 = affine.load %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.constant 0 : index
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %[[cst]]
  // CHECK-NEXT:   affine.load


  return
}

// -----

// CHECK-LABEL:   func @do_not_hoist_dependent_side_effect_free_op
func.func @do_not_hoist_dependent_side_effect_free_op(%arg0: memref<10x512xf32>) {
  %0 = memref.alloca() : memref<1xf32>
  %cst = arith.constant 8.0 : f32
  affine.for %i = 0 to 512 {
    affine.for %j = 0 to 10 {
      %5 = affine.load %arg0[%i, %j] : memref<10x512xf32>
      %6 = affine.load %0[0] : memref<1xf32>
      %add = arith.addf %5, %6 : f32
      affine.store %add, %0[0] : memref<1xf32>
    }
    %3 = affine.load %0[0] : memref<1xf32>
    %4 = arith.mulf %3, %cst : f32 // It shouldn't be hoisted.
  }
  return
}

// CHECK:       affine.for
// CHECK-NEXT:    affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      arith.addf
// CHECK-NEXT:      affine.store
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.load
// CHECK-NEXT:    arith.mulf
// CHECK-NEXT:  }

// -----

// CHECK-LABEL: func @vector_loop_nothing_invariant
func.func @vector_loop_nothing_invariant() {
  %m1 = memref.alloc() : memref<40xf32>
  %m2 = memref.alloc() : memref<40xf32>
  affine.for %arg0 = 0 to 10 {
    %v0 = affine.vector_load %m1[%arg0*4] : memref<40xf32>, vector<4xf32>
    %v1 = affine.vector_load %m2[%arg0*4] : memref<40xf32>, vector<4xf32>
    %v2 = arith.addf %v0, %v1 : vector<4xf32>
    affine.vector_store %v2, %m1[%arg0*4] : memref<40xf32>, vector<4xf32>
  }
  return
}

// CHECK:       affine.for
// CHECK-NEXT:    affine.vector_load
// CHECK-NEXT:    affine.vector_load
// CHECK-NEXT:    arith.addf
// CHECK-NEXT:    affine.vector_store
// CHECK-NEXT:  }

// -----

// CHECK-LABEL: func @vector_loop_all_invariant
func.func @vector_loop_all_invariant() {
  %m1 = memref.alloc() : memref<4xf32>
  %m2 = memref.alloc() : memref<4xf32>
  %m3 = memref.alloc() : memref<4xf32>
  affine.for %arg0 = 0 to 10 {
    %v0 = affine.vector_load %m1[0] : memref<4xf32>, vector<4xf32>
    %v1 = affine.vector_load %m2[0] : memref<4xf32>, vector<4xf32>
    %v2 = arith.addf %v0, %v1 : vector<4xf32>
    affine.vector_store %v2, %m3[0] : memref<4xf32>, vector<4xf32>
  }
  return
}

// CHECK:       memref.alloc()
// CHECK-NEXT:  memref.alloc()
// CHECK-NEXT:  memref.alloc()
// CHECK-NEXT:  affine.vector_load
// CHECK-NEXT:  affine.vector_load
// CHECK-NEXT:  arith.addf
// CHECK-NEXT:  affine.vector_store
// CHECK-NEXT:  affine.for

// -----

#set = affine_set<(d0): (d0 - 10 >= 0)>
// CHECK-LABEL:   func @affine_if_not_invariant(
func.func @affine_if_not_invariant(%buffer: memref<1024xf32>) -> f32 {
  %sum_init_0 = arith.constant 0.0 : f32
  %sum_init_1 = arith.constant 1.0 : f32
  %res = affine.for %i = 0 to 10 step 2 iter_args(%sum_iter = %sum_init_0) -> f32 {
    %t = affine.load %buffer[%i] : memref<1024xf32>
    %sum_next = affine.if #set(%i) -> (f32) {
      %new_sum = arith.addf %sum_iter, %t : f32
      affine.yield %new_sum : f32
    } else {
      affine.yield %sum_iter : f32
    }
    %modified_sum = arith.addf %sum_next, %sum_init_1 : f32
    affine.yield %modified_sum : f32
  }
  return %res : f32
}

// CHECK:       arith.constant 0.000000e+00 : f32
// CHECK-NEXT:  arith.constant 1.000000e+00 : f32
// CHECK-NEXT:  affine.for
// CHECK-NEXT:  affine.load
// CHECK-NEXT:  affine.if
// CHECK-NEXT:  arith.addf
// CHECK-NEXT:  affine.yield
// CHECK-NEXT:  } else {
// CHECK-NEXT:  affine.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  arith.addf
// CHECK-NEXT:  affine.yield
// CHECK-NEXT:  }

// -----

// CHECK-LABEL:   func @affine_for_not_invariant(
func.func @affine_for_not_invariant(%in : memref<30x512xf32, 1>,
                               %out : memref<30x1xf32, 1>) {
  %sum_0 = arith.constant 0.0 : f32
  %cst_0 = arith.constant 1.1 : f32
  affine.for %j = 0 to 30 {
    %sum = affine.for %i = 0 to 512 iter_args(%sum_iter = %sum_0) -> (f32) {
      %t = affine.load %in[%j,%i] : memref<30x512xf32,1>
      %sum_next = arith.addf %sum_iter, %t : f32
      affine.yield %sum_next : f32
    }
    %mod_sum = arith.mulf %sum, %cst_0 : f32
    affine.store %mod_sum, %out[%j, 0] : memref<30x1xf32, 1>
  }
  return
}

// CHECK:       arith.constant 0.000000e+00 : f32
// CHECK-NEXT:  arith.constant 1.100000e+00 : f32
// CHECK-NEXT:  affine.for
// CHECK-NEXT:  affine.for
// CHECK-NEXT:  affine.load
// CHECK-NEXT:  arith.addf
// CHECK-NEXT:  affine.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  arith.mulf
// CHECK-NEXT:  affine.store

// -----

// CHECK-LABEL: func @use_of_iter_operands_invariant
func.func @use_of_iter_operands_invariant(%m : memref<10xindex>) {
  %sum_1 = arith.constant 0 : index
  %v0 = affine.for %arg1 = 0 to 11 iter_args (%prevAccum = %sum_1) -> index {
    %prod = arith.muli %sum_1, %sum_1 : index
    %newAccum = arith.addi %prevAccum, %prod : index
    affine.yield %newAccum : index
  }
  return
}

// CHECK:       constant
// CHECK-NEXT:  muli
// CHECK-NEXT:  affine.for
// CHECK-NEXT:    addi
// CHECK-NEXT:    affine.yield

// -----

// CHECK-LABEL: func @use_of_iter_args_not_invariant
func.func @use_of_iter_args_not_invariant(%m : memref<10xindex>) {
  %sum_1 = arith.constant 0 : index
  %v0 = affine.for %arg1 = 0 to 11 iter_args (%prevAccum = %sum_1) -> index {
    %newAccum = arith.addi %prevAccum, %sum_1 : index
    affine.yield %newAccum : index
  }
  return
}

// CHECK:       arith.constant
// CHECK-NEXT:  affine.for
// CHECK-NEXT:  arith.addi
// CHECK-NEXT:  affine.yield

#map = affine_map<(d0) -> (64, d0 * -64 + 1020)>
// CHECK-LABEL: func.func @affine_parallel
func.func @affine_parallel(%memref_8: memref<4090x2040xf32>, %x: index) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.parallel (%arg3) = (0) to (32) {
    affine.for %arg4 = 0 to 16 {
      affine.parallel (%arg5, %arg6) = (0, 0) to (min(128, 122), min(64, %arg3 * -64 + 2040)) {
        affine.for %arg7 = 0 to min #map(%arg4) {
          affine.store %cst, %memref_8[%arg5 + 3968, %arg6 + %arg3 * 64] : memref<4090x2040xf32>
        }
      }
    }
  }
  // CHECK:       affine.parallel
  // CHECK-NEXT:    affine.for
  // CHECK-NEXT:      affine.parallel
  // CHECK-NEXT:        affine.store
  // CHECK-NEXT:        affine.for

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.parallel (%arg3) = (%c0) to (%c32) step (%c1) {
    affine.for %arg4 = 0 to 16 {
      affine.parallel (%arg5, %arg6) = (0, 0) to (min(128, 122), min(64, %x * -64 + 2040)) {
        affine.for %arg7 = 0 to min #map(%arg4) {
          affine.store %cst, %memref_8[%arg5 + 3968, %arg6] : memref<4090x2040xf32>
        }
      }
    }
  }
  // CHECK:       scf.parallel
  // CHECK-NEXT:    affine.for
  // CHECK-NEXT:      affine.parallel
  // CHECK-NEXT:        affine.store
  // CHECK-NEXT:        affine.for

  affine.for %arg3 = 0 to 32 {
    affine.for %arg4 = 0 to 16 {
      affine.parallel (%arg5, %arg6) = (0, 0) to (min(128, 122), min(64, %arg3 * -64 + 2040)) {
        // Unknown region-holding op for this pass.
        scf.for %arg7 = %c0 to %x step %c1 {
          affine.store %cst, %memref_8[%arg5 + 3968, %arg6 + %arg3 * 64] : memref<4090x2040xf32>
        }
      }
    }
  }
  // CHECK:       affine.for
  // CHECK-NEXT:    affine.for
  // CHECK-NEXT:      affine.parallel
  // CHECK-NEXT:        scf.for
  // CHECK-NEXT:          affine.store

  return
}

// -----

// CHECK-LABEL: func.func @affine_invariant_use_after_dma
#map = affine_map<(d0) -> (d0 * 163840)>
func.func @affine_invariant_use_after_dma(%arg0: memref<10485760xi32>, %arg1: memref<1xi32>, %arg2: memref<10485760xi32>) {
  %c320 = arith.constant 320 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<0xi32, 2>
  %alloc_0 = memref.alloc() : memref<1xi32, 2>
  affine.for %arg3 = 0 to 64 {
    %0 = affine.apply #map(%arg3)
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<0xi32, 2>
    %alloc_2 = memref.alloc() : memref<320xi32, 2>
    affine.dma_start %arg0[%0], %alloc_2[%c0], %alloc_1[%c0], %c320 : memref<10485760xi32>, memref<320xi32, 2>, memref<0xi32, 2>
    affine.dma_start %arg1[%c0], %alloc_0[%c0], %alloc[%c0], %c1 : memref<1xi32>, memref<1xi32, 2>, memref<0xi32, 2>
    affine.dma_wait %alloc_1[%c0], %c320 : memref<0xi32, 2>
    affine.dma_wait %alloc[%c0], %c1 : memref<0xi32, 2>
    %1 = affine.apply #map(%arg3)
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<0xi32, 2>
    %alloc_4 = memref.alloc() : memref<320xi32, 2>
    affine.for %arg4 = 0 to 320 {
      %2 = affine.load %alloc_2[%arg4] : memref<320xi32, 2>
      %3 = affine.load %alloc_0[0] : memref<1xi32, 2>
      %4 = arith.addi %2, %3 : i32
      %5 = arith.addi %4, %2 : i32
      affine.store %5, %alloc_4[%arg4] : memref<320xi32, 2>
    }
    affine.dma_start %alloc_4[%c0], %arg2[%1], %alloc_3[%c0], %c320 : memref<320xi32, 2>, memref<10485760xi32>, memref<0xi32, 2>
    affine.dma_wait %alloc_3[%c0], %c320 : memref<0xi32, 2>
  }
  return
}
// CHECK: %[[zero:.*]] = arith.constant 0 : index
// CHECK: %[[scalar_mem:.*]] = memref.alloc() : memref<1xi32, 2>
// CHECK: affine.dma_start %arg1[%[[zero]]], %alloc_0[%[[zero]]], %alloc[%[[zero]]], %c1
// CHECK: affine.load %[[scalar_mem]][0]

// -----

// CHECK-LABEL: func @affine_prefetch_invariant
func.func @affine_prefetch_invariant() {
  %0 = memref.alloc() : memref<10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[%i0, %i1] : memref<10x10xf32>
      affine.prefetch %0[%i0, %i0], write, locality<0>, data : memref<10x10xf32>
    }
  }

  // CHECK:      memref.alloc() : memref<10x10xf32>
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.prefetch
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:     %{{.*}}  = affine.load %{{.*}}[%{{.*}}  : memref<10x10xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  return
}
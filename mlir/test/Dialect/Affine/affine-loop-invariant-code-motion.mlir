// RUN: mlir-opt %s -affine-loop-invariant-code-motion -split-input-file | FileCheck %s

func @nested_loops_both_having_invariant_code() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      affine.store %v0, %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>

  return
}

// -----

// The store-load forwarding can see through affine apply's since it relies on
// dependence information.
// CHECK-LABEL: func @store_affine_apply
func @store_affine_apply() -> memref<10xf32> {
  %cf7 = constant 7.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %arg0 = 0 to 10 {
      %t0 = affine.apply affine_map<(d1) -> (d1 + 1)>(%arg0)
      affine.store %cf7, %m[%t0] : memref<10xf32>
  }
  return %m : memref<10xf32>
// CHECK:       %cst = constant 7.000000e+00 : f32
// CHECK-NEXT:  %0 = memref.alloc() : memref<10xf32>
// CHECK-NEXT:  affine.for %arg0 = 0 to 10 {
// CHECK-NEXT:      %1 = affine.apply #map{{[0-9]*}}(%arg0)
// CHECK-NEXT:      affine.store %cst, %0[%1] : memref<10xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %0 : memref<10xf32>
}

// -----

func @nested_loops_code_invariant_to_both() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      %v0 = addf %cf7, %cf8 : f32
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32

  return
}

// -----

// CHECK-LABEL: func @nested_loops_inner_loops_invariant_to_outermost_loop
func @nested_loops_inner_loops_invariant_to_outermost_loop(%m : memref<10xindex>) {
  affine.for %arg0 = 0 to 20 {
    affine.for %arg1 = 0 to 30 {
      %v0 = affine.for %arg2 = 0 to 10 iter_args (%prevAccum = %arg1) -> index {
        %v1 = affine.load %m[%arg2] : memref<10xindex>
        %newAccum = addi %prevAccum, %v1 : index
        affine.yield %newAccum : index
      }
    }
  }

  // CHECK:      affine.for %{{.*}} = 0 to 30 {
  // CHECK-NEXT:   %{{.*}}  = affine.for %{{.*}}  = 0 to 10 iter_args(%{{.*}} = %{{.*}}) -> (index) {
  // CHECK-NEXT:     %{{.*}}  = affine.load %{{.*}}[%{{.*}}  : memref<10xindex>
  // CHECK-NEXT:     %{{.*}}  = addi %{{.*}}, %{{.*}} : index
  // CHECK-NEXT:     affine.yield %{{.*}} : index
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 20 {
  // CHECK-NEXT: }

  return
}

// -----

func @single_loop_nothing_invariant() {
  %m1 = memref.alloc() : memref<10xf32>
  %m2 = memref.alloc() : memref<10xf32>
  affine.for %arg0 = 0 to 10 {
    %v0 = affine.load %m1[%arg0] : memref<10xf32>
    %v1 = affine.load %m2[%arg0] : memref<10xf32>
    %v2 = addf %v0, %v1 : f32
    affine.store %v2, %m1[%arg0] : memref<10xf32>
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %1 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: %2 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: %3 = affine.load %1[%arg0] : memref<10xf32>
  // CHECK-NEXT: %4 = addf %2, %3 : f32
  // CHECK-NEXT: affine.store %4, %0[%arg0] : memref<10xf32>

  return
}

// -----

func @invariant_code_inside_affine_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %t0 = affine.apply affine_map<(d1) -> (d1 + 1)>(%arg0)
    affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %t0) {
        %cf9 = addf %cf8, %cf8 : f32
        affine.store %cf9, %m[%arg0] : memref<10xf32>

    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: %1 = affine.apply #map{{[0-9]*}}(%arg0)
  // CHECK-NEXT: affine.if #set(%arg0, %1) {
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %2, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: }


  return
}

// -----

func @dependent_stores() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = addf %cf7, %cf7 : f32
      affine.store %v1, %m[%arg1] : memref<10xf32>
      affine.store %v0, %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {

  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:   affine.store %2, %0[%arg1] : memref<10xf32>
  // CHECK-NEXT:   affine.store %1, %0[%arg0] : memref<10xf32>

  return
}

// -----

func @independent_stores() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = addf %cf7, %cf7 : f32
      affine.store %v0, %m[%arg0] : memref<10xf32>
      affine.store %v1, %m[%arg1] : memref<10xf32>
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT:   affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:     affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT:     affine.store %2, %0[%arg1] : memref<10xf32>
  // CHECK-NEXT:    }

  return
}

// -----

func @load_dependent_store() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = addf %cf7, %cf7 : f32
      affine.store %v0, %m[%arg1] : memref<10xf32>
      %v2 = affine.load %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:   affine.store %1, %0[%arg1] : memref<10xf32>
  // CHECK-NEXT:   %3 = affine.load %0[%arg0] : memref<10xf32>

  return
}

// -----

func @load_after_load() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = addf %cf7, %cf7 : f32
      %v3 = affine.load %m[%arg1] : memref<10xf32>
      %v2 = affine.load %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: %3 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>

  return
}

// -----

func @invariant_affine_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>

      }
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: }


  return
}

// -----

func @invariant_affine_if2() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg1] : memref<10xf32>

      }
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %1, %0[%arg1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func @invariant_affine_nested_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>
          affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
            affine.store %cf9, %m[%arg1] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: affine.store %1, %0[%arg1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func @invariant_affine_nested_if_else() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>
          affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
            affine.store %cf9, %m[%arg0] : memref<10xf32>
          } else {
            affine.store %cf9, %m[%arg1] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: } else {
  // CHECK-NEXT: affine.store %1, %0[%arg1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func @invariant_affine_nested_if_else2() {
  %m = memref.alloc() : memref<10xf32>
  %m2 = memref.alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          %tload1 = affine.load %m[%arg0] : memref<10xf32>
          affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
            affine.store %cf9, %m2[%arg0] : memref<10xf32>
          } else {
            %tload2 = affine.load %m[%arg0] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %1 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: %3 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: affine.store %2, %1[%arg0] : memref<10xf32>
  // CHECK-NEXT: } else {
  // CHECK-NEXT: %4 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func @invariant_affine_nested_if2() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          %v1 = affine.load %m[%arg0] : memref<10xf32>
          affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
            %v2 = affine.load %m[%arg0] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: %2 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: %3 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func @invariant_affine_for_inside_affine_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>
          affine.for %arg2 = 0 to 10 {
            affine.store %cf9, %m[%arg2] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT: affine.if #set(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.for %arg2 = 0 to 10 {
  // CHECK-NEXT: affine.store %1, %0[%arg2] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func @invariant_constant_and_load() {
  %m = memref.alloc() : memref<100xf32>
  %m2 = memref.alloc() : memref<100xf32>
  affine.for %arg0 = 0 to 5 {
    %c0 = constant 0 : index
    %v = affine.load %m2[%c0] : memref<100xf32>
    affine.store %v, %m[%arg0] : memref<100xf32>
  }

  // CHECK: %0 = memref.alloc() : memref<100xf32>
  // CHECK-NEXT: %1 = memref.alloc() : memref<100xf32>
  // CHECK-NEXT: %c0 = constant 0 : index
  // CHECK-NEXT: %2 = affine.load %1[%c0] : memref<100xf32>
  // CHECK-NEXT: affine.for %arg0 = 0 to 5 {
  // CHECK-NEXT:  affine.store %2, %0[%arg0] : memref<100xf32>


  return
}

// -----

func @nested_load_store_same_memref() {
  %m = memref.alloc() : memref<10xf32>
  %cst = constant 8.0 : f32
  %c0 = constant 0 : index
   affine.for %arg0 = 0 to 10 {
    %v0 = affine.load %m[%c0] : memref<10xf32>
    affine.for %arg1 = 0 to 10 {
      affine.store %cst, %m[%arg1] : memref<10xf32>
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: %c0 = constant 0 : index
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT:  %1 = affine.load %0[%c0] : memref<10xf32>
  // CHECK-NEXT:   affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:    affine.store %cst, %0[%arg1] : memref<10xf32>


  return
}

// -----

func @nested_load_store_same_memref2() {
  %m = memref.alloc() : memref<10xf32>
  %cst = constant 8.0 : f32
  %c0 = constant 0 : index
   affine.for %arg0 = 0 to 10 {
     affine.store %cst, %m[%c0] : memref<10xf32>
      affine.for %arg1 = 0 to 10 {
        %v0 = affine.load %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: %c0 = constant 0 : index
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %cst, %0[%c0] : memref<10xf32>
  // CHECK-NEXT:   %1 = affine.load %0[%arg0] : memref<10xf32>


  return
}

// -----

// CHECK-LABEL:   func @do_not_hoist_dependent_side_effect_free_op
func @do_not_hoist_dependent_side_effect_free_op(%arg0: memref<10x512xf32>) {
  %0 = memref.alloca() : memref<1xf32>
  %cst = constant 8.0 : f32
  affine.for %i = 0 to 512 {
    affine.for %j = 0 to 10 {
      %5 = affine.load %arg0[%i, %j] : memref<10x512xf32>
      %6 = affine.load %0[0] : memref<1xf32>
      %add = addf %5, %6 : f32
      affine.store %add, %0[0] : memref<1xf32>
    }
    %3 = affine.load %0[0] : memref<1xf32>
    %4 = mulf %3, %cst : f32 // It shouldn't be hoisted.
  }
  return
}

// CHECK:       affine.for
// CHECK-NEXT:    affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      addf
// CHECK-NEXT:      affine.store
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.load
// CHECK-NEXT:    mulf
// CHECK-NEXT:  }

// -----

// CHECK-LABEL: func @vector_loop_nothing_invariant
func @vector_loop_nothing_invariant() {
  %m1 = memref.alloc() : memref<40xf32>
  %m2 = memref.alloc() : memref<40xf32>
  affine.for %arg0 = 0 to 10 {
    %v0 = affine.vector_load %m1[%arg0*4] : memref<40xf32>, vector<4xf32>
    %v1 = affine.vector_load %m2[%arg0*4] : memref<40xf32>, vector<4xf32>
    %v2 = addf %v0, %v1 : vector<4xf32>
    affine.vector_store %v2, %m1[%arg0*4] : memref<40xf32>, vector<4xf32>
  }
  return
}

// CHECK:       affine.for
// CHECK-NEXT:    affine.vector_load
// CHECK-NEXT:    affine.vector_load
// CHECK-NEXT:    addf
// CHECK-NEXT:    affine.vector_store
// CHECK-NEXT:  }

// -----

// CHECK-LABEL: func @vector_loop_all_invariant
func @vector_loop_all_invariant() {
  %m1 = memref.alloc() : memref<4xf32>
  %m2 = memref.alloc() : memref<4xf32>
  %m3 = memref.alloc() : memref<4xf32>
  affine.for %arg0 = 0 to 10 {
    %v0 = affine.vector_load %m1[0] : memref<4xf32>, vector<4xf32>
    %v1 = affine.vector_load %m2[0] : memref<4xf32>, vector<4xf32>
    %v2 = addf %v0, %v1 : vector<4xf32>
    affine.vector_store %v2, %m3[0] : memref<4xf32>, vector<4xf32>
  }
  return
}

// CHECK:       memref.alloc()
// CHECK-NEXT:  memref.alloc()
// CHECK-NEXT:  memref.alloc()
// CHECK-NEXT:  affine.vector_load
// CHECK-NEXT:  affine.vector_load
// CHECK-NEXT:  addf
// CHECK-NEXT:  affine.vector_store
// CHECK-NEXT:  affine.for

// -----

#set = affine_set<(d0): (d0 - 10 >= 0)>
// CHECK-LABEL:   func @affine_if_not_invariant(
func @affine_if_not_invariant(%buffer: memref<1024xf32>) -> f32 {
  %sum_init_0 = constant 0.0 : f32
  %sum_init_1 = constant 1.0 : f32
  %res = affine.for %i = 0 to 10 step 2 iter_args(%sum_iter = %sum_init_0) -> f32 {
    %t = affine.load %buffer[%i] : memref<1024xf32>
    %sum_next = affine.if #set(%i) -> (f32) {
      %new_sum = addf %sum_iter, %t : f32
      affine.yield %new_sum : f32
    } else {
      affine.yield %sum_iter : f32
    }
    %modified_sum = addf %sum_next, %sum_init_1 : f32
    affine.yield %modified_sum : f32
  }
  return %res : f32
}

// CHECK:       constant 0.000000e+00 : f32
// CHECK-NEXT:  constant 1.000000e+00 : f32
// CHECK-NEXT:  affine.for
// CHECK-NEXT:  affine.load
// CHECK-NEXT:  affine.if
// CHECK-NEXT:  addf
// CHECK-NEXT:  affine.yield
// CHECK-NEXT:  } else {
// CHECK-NEXT:  affine.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  addf
// CHECK-NEXT:  affine.yield
// CHECK-NEXT:  }

// -----

// CHECK-LABEL:   func @affine_for_not_invariant(
func @affine_for_not_invariant(%in : memref<30x512xf32, 1>,
                               %out : memref<30x1xf32, 1>) {
  %sum_0 = constant 0.0 : f32
  %cst_0 = constant 1.1 : f32
  affine.for %j = 0 to 30 {
    %sum = affine.for %i = 0 to 512 iter_args(%sum_iter = %sum_0) -> (f32) {
      %t = affine.load %in[%j,%i] : memref<30x512xf32,1>
      %sum_next = addf %sum_iter, %t : f32
      affine.yield %sum_next : f32
    }
    %mod_sum = mulf %sum, %cst_0 : f32
    affine.store %mod_sum, %out[%j, 0] : memref<30x1xf32, 1>
  }
  return
}

// CHECK:       constant 0.000000e+00 : f32
// CHECK-NEXT:  constant 1.100000e+00 : f32
// CHECK-NEXT:  affine.for
// CHECK-NEXT:  affine.for
// CHECK-NEXT:  affine.load
// CHECK-NEXT:  addf
// CHECK-NEXT:  affine.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  mulf
// CHECK-NEXT:  affine.store

// -----

// CHECK-LABEL: func @use_of_iter_args_not_invariant
func @use_of_iter_args_not_invariant(%m : memref<10xindex>) {
  %sum_1 = constant 0 : index
  %v0 = affine.for %arg1 = 0 to 11 iter_args (%prevAccum = %sum_1) -> index {
    %newAccum = addi %prevAccum, %sum_1 : index
    affine.yield %newAccum : index
  }
  return
}

// CHECK:       constant
// CHECK-NEXT:  affine.for
// CHECK-NEXT:  addi
// CHECK-NEXT:  affine.yield

// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// -----

func @affine_apply_operand_non_index(%arg0 : i32) {
  // Custom parser automatically assigns all arguments the `index` so we must
  // use the generic syntax here to exercise the verifier.
  // expected-error@+1 {{op operand #0 must be index, but got 'i32'}}
  %0 = "affine.apply"(%arg0) {map = affine_map<(d0) -> (d0)>} : (i32) -> (index)
  return
}

// -----

func @affine_apply_resul_non_index(%arg0 : index) {
  // Custom parser automatically assigns `index` as the result type so we must
  // use the generic syntax here to exercise the verifier.
  // expected-error@+1 {{op result #0 must be index, but got 'i32'}}
  %0 = "affine.apply"(%arg0) {map = affine_map<(d0) -> (d0)>} : (index) -> (i32)
  return
}

// -----

#map = affine_map<(d0)[s0] -> (d0 + s0)>

func @affine_for_lower_bound_invalid_dim(%arg : index) {
  affine.for %n0 = 0 to 7 {
    %dim = addi %arg, %arg : index

    // expected-error@+1 {{operand cannot be used as a dimension id}}
    affine.for %n1 = 0 to #map(%dim)[%arg] {
    }
  }
  return
}

// -----

#map = affine_map<(d0)[s0] -> (d0 + s0)>

func @affine_for_upper_bound_invalid_dim(%arg : index) {
  affine.for %n0 = 0 to 7 {
    %dim = addi %arg, %arg : index

    // expected-error@+1 {{operand cannot be used as a dimension id}}
    affine.for %n1 = #map(%dim)[%arg] to 7 {
    }
  }
  return
}

// -----
func @affine_load_invalid_dim(%M : memref<10xi32>) {
  "unknown"() ({
  ^bb0(%arg: index):
    affine.load %M[%arg] : memref<10xi32>
    // expected-error@-1 {{index must be a dimension or symbol identifier}}
    br ^bb1
  ^bb1:
    br ^bb1
  }) : () -> ()
  return
}

// -----

#map0 = affine_map<(d0)[s0] -> (d0 + s0)>

func @affine_for_lower_bound_invalid_sym() {
  affine.for %i0 = 0 to 7 {
    // expected-error@+1 {{operand cannot be used as a symbol}}
    affine.for %n0 = #map0(%i0)[%i0] to 7 {
    }
  }
  return
}

// -----

#map0 = affine_map<(d0)[s0] -> (d0 + s0)>

func @affine_for_upper_bound_invalid_sym() {
  affine.for %i0 = 0 to 7 {
    // expected-error@+1 {{operand cannot be used as a symbol}}
    affine.for %n0 = 0 to #map0(%i0)[%i0] {
    }
  }
  return
}

// -----

#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func @affine_if_invalid_dim(%arg : index) {
  affine.for %n0 = 0 to 7 {
    %dim = addi %arg, %arg : index

    // expected-error@+1 {{operand cannot be used as a dimension id}}
    affine.if #set0(%dim)[%n0] {}
  }
  return
}

// -----

#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func @affine_if_invalid_sym() {
  affine.for %i0 = 0 to 7 {
    // expected-error@+1 {{operand cannot be used as a symbol}}
    affine.if #set0(%i0)[%i0] {}
  }
  return
}

// -----

#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func @affine_if_invalid_dimop_dim(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  affine.for %n0 = 0 to 7 {
    %0 = alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    %dim = dim %0, 0 : memref<?x?x?x?xf32>

    // expected-error@+1 {{operand cannot be used as a dimension id}}
    affine.if #set0(%dim)[%n0] {}
  }
  return
}

// -----

func @affine_store_missing_l_square(%C: memref<4096x4096xf32>) {
  %9 = constant 0.0 : f32
  // expected-error@+1 {{expected '['}}
  affine.store %9, %C : memref<4096x4096xf32>
  return
}

// -----

// CHECK-LABEL: @affine_min
func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.min affine_map<(d0) -> (d0)> (%arg0, %arg1)

  return
}

// -----

// CHECK-LABEL: @affine_min
func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.min affine_map<()[s0] -> (s0)> (%arg0, %arg1)

  return
}

// -----

// CHECK-LABEL: @affine_min
func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.min affine_map<(d0) -> (d0)> ()

  return
}

// -----

// CHECK-LABEL: @affine_max
func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.max affine_map<(d0) -> (d0)> (%arg0, %arg1)

  return
}

// -----

// CHECK-LABEL: @affine_max
func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.max affine_map<()[s0] -> (s0)> (%arg0, %arg1)

  return
}

// -----

// CHECK-LABEL: @affine_max
func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.max affine_map<(d0) -> (d0)> ()

  return
}

// -----

// CHECK-LABEL: @affine_parallel
func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{region argument count and num results of upper bounds, lower bounds, and steps must all match}}
  affine.parallel (%i) = (0, 0) to (100, 100) step (10, 10) {
  }
}

// -----

// CHECK-LABEL: @affine_parallel
func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{region argument count and num results of upper bounds, lower bounds, and steps must all match}}
  affine.parallel (%i, %j) = (0) to (100, 100) step (10, 10) {
  }
}

// -----

// CHECK-LABEL: @affine_parallel
func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{region argument count and num results of upper bounds, lower bounds, and steps must all match}}
  affine.parallel (%i, %j) = (0, 0) to (100) step (10, 10) {
  }
}

// -----

// CHECK-LABEL: @affine_parallel
func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{region argument count and num results of upper bounds, lower bounds, and steps must all match}}
  affine.parallel (%i, %j) = (0, 0) to (100, 100) step (10) {
  }
}

// -----

// CHECK-LABEL: @affine_parallel

func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  affine.for %x = 0 to 7 {
    %y = addi %x, %x : index
    // expected-error@+1 {{operand cannot be used as a dimension id}}
    affine.parallel (%i, %j) = (0, 0) to (%y, 100) step (10, 10) {
    }
  }
  return
}

// -----

// CHECK-LABEL: @affine_parallel

func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  affine.for %x = 0 to 7 {
    %y = addi %x, %x : index
    // expected-error@+1 {{operand cannot be used as a symbol}}
    affine.parallel (%i, %j) = (0, 0) to (symbol(%y), 100) step (10, 10) {
    }
  }
  return
}

// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// -----

func.func @affine_apply_operand_non_index(%arg0 : i32) {
  // Custom parser automatically assigns all arguments the `index` so we must
  // use the generic syntax here to exercise the verifier.
  // expected-error@+1 {{op operand #0 must be index, but got 'i32'}}
  %0 = "affine.apply"(%arg0) {map = affine_map<(d0) -> (d0)>} : (i32) -> (index)
  return
}

// -----

func.func @affine_apply_resul_non_index(%arg0 : index) {
  // Custom parser automatically assigns `index` as the result type so we must
  // use the generic syntax here to exercise the verifier.
  // expected-error@+1 {{op result #0 must be index, but got 'i32'}}
  %0 = "affine.apply"(%arg0) {map = affine_map<(d0) -> (d0)>} : (index) -> (i32)
  return
}

// -----

#map = affine_map<(d0)[s0] -> (d0 + s0)>

func.func @affine_for_lower_bound_invalid_dim(%arg : index) {
  affine.for %n0 = 0 to 7 {
    %dim = arith.addi %arg, %arg : index

    // expected-error@+1 {{operand cannot be used as a dimension id}}
    affine.for %n1 = 0 to #map(%dim)[%arg] {
    }
  }
  return
}

// -----

#map = affine_map<(d0)[s0] -> (d0 + s0)>

func.func @affine_for_upper_bound_invalid_dim(%arg : index) {
  affine.for %n0 = 0 to 7 {
    %dim = arith.addi %arg, %arg : index

    // expected-error@+1 {{operand cannot be used as a dimension id}}
    affine.for %n1 = #map(%dim)[%arg] to 7 {
    }
  }
  return
}

// -----
func.func @affine_load_invalid_dim(%M : memref<10xi32>) {
  "unknown"() ({
  ^bb0(%arg: index):
    affine.load %M[%arg] : memref<10xi32>
    // expected-error@-1 {{index must be a dimension or symbol identifier}}
    cf.br ^bb1
  ^bb1:
    cf.br ^bb1
  }) : () -> ()
  return
}

// -----

#map0 = affine_map<(d0)[s0] -> (d0 + s0)>

func.func @affine_for_lower_bound_invalid_sym() {
  affine.for %i0 = 0 to 7 {
    // expected-error@+1 {{operand cannot be used as a symbol}}
    affine.for %n0 = #map0(%i0)[%i0] to 7 {
    }
  }
  return
}

// -----

#map0 = affine_map<(d0)[s0] -> (d0 + s0)>

func.func @affine_for_upper_bound_invalid_sym() {
  affine.for %i0 = 0 to 7 {
    // expected-error@+1 {{operand cannot be used as a symbol}}
    affine.for %n0 = 0 to #map0(%i0)[%i0] {
    }
  }
  return
}

// -----

#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func.func @affine_if_invalid_dim(%arg : index) {
  affine.for %n0 = 0 to 7 {
    %dim = arith.addi %arg, %arg : index

    // expected-error@+1 {{operand cannot be used as a dimension id}}
    affine.if #set0(%dim)[%n0] {}
  }
  return
}

// -----

#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func.func @affine_if_invalid_sym() {
  affine.for %i0 = 0 to 7 {
    // expected-error@+1 {{operand cannot be used as a symbol}}
    affine.if #set0(%i0)[%i0] {}
  }
  return
}

// -----

#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func.func @affine_if_invalid_dimop_dim(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  affine.for %n0 = 0 to 7 {
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %0, %c0 : memref<?x?x?x?xf32>

    // expected-error@+1 {{operand cannot be used as a symbol}}
    affine.if #set0(%dim)[%n0] {}
  }
  return
}

// -----

func.func @affine_store_missing_l_square(%C: memref<4096x4096xf32>) {
  %9 = arith.constant 0.0 : f32
  // expected-error@+1 {{expected '['}}
  affine.store %9, %C : memref<4096x4096xf32>
  return
}

// -----

func.func @affine_store_wrong_value_type(%C: memref<f32>) {
  %c0 = arith.constant 0 : i32
  // expected-error@+1 {{value to store must have the same type as memref element type}}
  "affine.store"(%c0, %C) : (i32, memref<f32>) -> ()
  return
}

// -----

func.func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.min affine_map<(d0) -> (d0)> (%arg0, %arg1)

  return
}

// -----

func.func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.min affine_map<()[s0] -> (s0)> (%arg0, %arg1)

  return
}

// -----

func.func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.min affine_map<(d0) -> (d0)> ()

  return
}

// -----

func.func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.max affine_map<(d0) -> (d0)> (%arg0, %arg1)

  return
}

// -----

func.func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.max affine_map<()[s0] -> (s0)> (%arg0, %arg1)

  return
}

// -----

func.func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{operand count and affine map dimension and symbol count must match}}
  %0 = affine.max affine_map<(d0) -> (d0)> ()

  return
}

// -----

func.func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{the number of region arguments (1) and the number of map groups for lower (2) and upper bound (2), and the number of steps (2) must all match}}
  affine.parallel (%i) = (0, 0) to (100, 100) step (10, 10) {
  }
}

// -----

func.func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{the number of region arguments (2) and the number of map groups for lower (1) and upper bound (2), and the number of steps (2) must all match}}
  affine.parallel (%i, %j) = (0) to (100, 100) step (10, 10) {
  }
}

// -----

func.func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{the number of region arguments (2) and the number of map groups for lower (2) and upper bound (1), and the number of steps (2) must all match}}
  affine.parallel (%i, %j) = (0, 0) to (100) step (10, 10) {
  }
}

// -----

func.func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{the number of region arguments (2) and the number of map groups for lower (2) and upper bound (2), and the number of steps (1) must all match}}
  affine.parallel (%i, %j) = (0, 0) to (100, 100) step (10) {
  }
}

// -----

func.func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  affine.for %x = 0 to 7 {
    %y = arith.addi %x, %x : index
    // expected-error@+1 {{operand cannot be used as a dimension id}}
    affine.parallel (%i, %j) = (0, 0) to (%y, 100) step (10, 10) {
    }
  }
  return
}

// -----

func.func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  affine.for %x = 0 to 7 {
    %y = arith.addi %x, %x : index
    // expected-error@+1 {{operand cannot be used as a symbol}}
    affine.parallel (%i, %j) = (0, 0) to (symbol(%y), 100) step (10, 10) {
    }
  }
  return
}

// -----

func.func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<100x100xf32>
  //  expected-error@+1 {{reduction must be specified for each output}}
  %1 = affine.parallel (%i, %j) = (0, 0) to (100, 100) step (10, 10) -> (f32) {
    %2 = affine.load %0[%i, %j] : memref<100x100xf32>
    affine.yield %2 : f32
  }
  return
}

// -----

func.func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<100x100xf32>
  //  expected-error@+1 {{invalid reduction value: "bad"}}
  %1 = affine.parallel (%i, %j) = (0, 0) to (100, 100) step (10, 10) reduce ("bad") -> (f32) {
    %2 = affine.load %0[%i, %j] : memref<100x100xf32>
    affine.yield %2 : f32
  }
  return
}

// -----

func.func @affine_parallel(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<100x100xi32>
  %1 = affine.parallel (%i, %j) = (0, 0) to (100, 100) step (10, 10) reduce ("minf") -> (f32) {
    %2 = affine.load %0[%i, %j] : memref<100x100xi32>
    //  expected-error@+1 {{types mismatch between yield op and its parent}}
    affine.yield %2 : i32
  }
  return
}

// -----

func.func @vector_load_invalid_vector_type() {
  %0 = memref.alloc() : memref<100xf32>
  affine.for %i0 = 0 to 16 step 8 {
    // expected-error@+1 {{requires memref and vector types of the same elemental type}}
    %1 = affine.vector_load %0[%i0] : memref<100xf32>, vector<8xf64>
  }
  return
}

// -----

func.func @vector_store_invalid_vector_type() {
  %0 = memref.alloc() : memref<100xf32>
  %1 = arith.constant dense<7.0> : vector<8xf64>
  affine.for %i0 = 0 to 16 step 8 {
    // expected-error@+1 {{requires memref and vector types of the same elemental type}}
    affine.vector_store %1, %0[%i0] : memref<100xf32>, vector<8xf64>
  }
  return
}

// -----

func.func @vector_load_vector_memref() {
  %0 = memref.alloc() : memref<100xvector<8xf32>>
  affine.for %i0 = 0 to 4 {
    // expected-error@+1 {{requires memref and vector types of the same elemental type}}
    %1 = affine.vector_load %0[%i0] : memref<100xvector<8xf32>>, vector<8xf32>
  }
  return
}

// -----

func.func @vector_store_vector_memref() {
  %0 = memref.alloc() : memref<100xvector<8xf32>>
  %1 = arith.constant dense<7.0> : vector<8xf32>
  affine.for %i0 = 0 to 4 {
    // expected-error@+1 {{requires memref and vector types of the same elemental type}}
    affine.vector_store %1, %0[%i0] : memref<100xvector<8xf32>>, vector<8xf32>
  }
  return
}

// -----

func.func @affine_if_with_then_region_args(%N: index) {
  %c = arith.constant 200 : index
  %i = arith.constant 20: index
  // expected-error@+1 {{affine.if' op region #0 should have no arguments}}
  affine.if affine_set<(i)[N] : (i - 2 >= 0, 4 - i >= 0)>(%i)[%c]  {
    ^bb0(%arg:i32):
      %w = affine.apply affine_map<(d0,d1)[s0] -> (d0+d1+s0)> (%i, %i) [%N]
  }
  return
}

// -----

func.func @affine_if_with_else_region_args(%N: index) {
  %c = arith.constant 200 : index
  %i = arith.constant 20: index
  // expected-error@+1 {{affine.if' op region #1 should have no arguments}}
  affine.if affine_set<(i)[N] : (i - 2 >= 0, 4 - i >= 0)>(%i)[%c]  {
      %w = affine.apply affine_map<(d0,d1)[s0] -> (d0+d1+s0)> (%i, %i) [%N]
  } else {
    ^bb0(%arg:i32):
      %w = affine.apply affine_map<(d0,d1)[s0] -> (d0-d1+s0)> (%i, %i) [%N]
  }
  return
}

// -----

func.func @affine_for_iter_args_mismatch(%buffer: memref<1024xf32>) -> f32 {
  %sum_0 = arith.constant 0.0 : f32
  // expected-error@+1 {{mismatch between the number of loop-carried values and results}}
  %res = affine.for %i = 0 to 10 step 2 iter_args(%sum_iter = %sum_0) -> (f32, f32) {
    %t = affine.load %buffer[%i] : memref<1024xf32>
    affine.yield %t : f32
  }
  return %res : f32
}


// -----

func.func @result_number() {
  // expected-error@+1 {{result number not allowed}}
  affine.for %n0#0 = 0 to 7 {
  }
  return
}

// -----

func.func @malformed_for_percent() {
  affine.for i = 1 to 10 { // expected-error {{expected SSA operand}}

// -----

func.func @malformed_for_equal() {
  affine.for %i 1 to 10 { // expected-error {{expected '='}}

// -----

func.func @malformed_for_to() {
  affine.for %i = 1 too 10 { // expected-error {{expected 'to' between bounds}}
  }
}

// -----

func.func @incomplete_for() {
  affine.for %i = 1 to 10 step 2
}        // expected-error @-1 {{expected '{' to begin a region}}

// -----

#map0 = affine_map<(d0) -> (d0 floordiv 4)>

func.func @reference_to_iv_in_bound() {
  // expected-error@+2 {{region entry argument '%i0' is already in use}}
  // expected-note@+1 {{previously referenced here}}
  affine.for %i0 = #map0(%i0) to 10 {
  }
}

// -----

func.func @nonconstant_step(%1 : i32) {
  affine.for %2 = 1 to 5 step %1 { // expected-error {{expected attribute value}}

// -----

func.func @for_negative_stride() {
  affine.for %i = 1 to 10 step -1
}        // expected-error@-1 {{expected step to be representable as a positive signed integer}}

// -----

func.func @invalid_if_conditional2() {
  affine.for %i = 1 to 10 {
    affine.if affine_set<(i)[N] : (i >= )>  // expected-error {{expected affine expression}}
  }
}

// -----

func.func @invalid_if_conditional3() {
  affine.for %i = 1 to 10 {
    affine.if affine_set<(i)[N] : (i == )>  // expected-error {{expected affine expression}}
  }
}

// -----

func.func @invalid_if_conditional6() {
  affine.for %i = 1 to 10 {
    affine.if affine_set<(i) : (i)> // expected-error {{expected '== affine-expr' or '>= affine-expr' at end of affine constraint}}
  }
}

// -----
// TODO: support affine.if (1)?
func.func @invalid_if_conditional7() {
  affine.for %i = 1 to 10 {
    affine.if affine_set<(i) : (1)> // expected-error {{expected '== affine-expr' or '>= affine-expr' at end of affine constraint}}
  }
}

// -----

func.func @missing_for_max(%arg0: index, %arg1: index, %arg2: memref<100xf32>) {
  // expected-error @+1 {{lower loop bound affine map with multiple results requires 'max' prefix}}
  affine.for %i0 = affine_map<()[s]->(0,s-1)>()[%arg0] to %arg1 {
  }
  return
}

// -----

func.func @missing_for_min(%arg0: index, %arg1: index, %arg2: memref<100xf32>) {
  // expected-error @+1 {{upper loop bound affine map with multiple results requires 'min' prefix}}
  affine.for %i0 = %arg0 to affine_map<()[s]->(100,s+1)>()[%arg1] {
  }
  return
}

// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -simplify-affine-structures | FileCheck %s

// CHECK-DAG: #[[$SET_EMPTY:.*]] = affine_set<() : (1 == 0)>
// CHECK-DAG: #[[$SET_2D:.*]] = affine_set<(d0, d1) : (d0 - 100 == 0, d1 - 10 == 0, -d0 + 100 >= 0, d1 >= 0)>
// CHECK-DAG: #[[$SET_7_11:.*]] = affine_set<(d0, d1) : (d0 * 7 + d1 * 5 + 88 == 0, d0 * 5 - d1 * 11 + 60 == 0, d0 * 11 + d1 * 7 - 24 == 0, d0 * 7 + d1 * 5 + 88 == 0)>

// An external function that we will use in bodies to avoid DCE.
func private @external() -> ()

// CHECK-LABEL: func @test_gaussian_elimination_empty_set0() {
func @test_gaussian_elimination_empty_set0() {
  affine.for %arg0 = 1 to 10 {
    affine.for %arg1 = 1 to 100 {
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0, d1) : (2 == 0)>(%arg0, %arg1) {
        call @external() : () -> ()
      }
    }
  }
  return
}

// CHECK-LABEL: func @test_gaussian_elimination_empty_set1() {
func @test_gaussian_elimination_empty_set1() {
  affine.for %arg0 = 1 to 10 {
    affine.for %arg1 = 1 to 100 {
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0, d1) : (1 >= 0, -1 >= 0)> (%arg0, %arg1) {
        call @external() : () -> ()
      }
    }
  }
  return
}

// CHECK-LABEL: func @test_gaussian_elimination_non_empty_set2() {
func @test_gaussian_elimination_non_empty_set2() {
  affine.for %arg0 = 1 to 10 {
    affine.for %arg1 = 1 to 100 {
      // CHECK: #[[$SET_2D]](%arg0, %arg1)
      affine.if affine_set<(d0, d1) : (d0 - 100 == 0, d1 - 10 == 0, -d0 + 100 >= 0, d1 >= 0, d1 + 101 >= 0)>(%arg0, %arg1) {
        call @external() : () -> ()
      }
    }
  }
  return
}

// CHECK-LABEL: func @test_gaussian_elimination_empty_set3() {
func @test_gaussian_elimination_empty_set3() {
  %c7 = constant 7 : index
  %c11 = constant 11 : index
  affine.for %arg0 = 1 to 10 {
    affine.for %arg1 = 1 to 100 {
      // CHECK: #[[$SET_EMPTY]]()
      affine.if affine_set<(d0, d1)[s0, s1] : (d0 - s0 == 0, d0 + s0 == 0, s0 - 1 == 0)>(%arg0, %arg1)[%c7, %c11] {
        call @external() : () -> ()
      }
    }
  }
  return
}

// Set for test case: test_gaussian_elimination_non_empty_set4
#set_2d_non_empty = affine_set<(d0, d1)[s0, s1] : (d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0,
                                       d0 * 5 - d1 * 11 + s0 * 7 + s1 == 0,
                                       d0 * 11 + d1 * 7 - s0 * 5 + s1 == 0,
                                       d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0)>

// CHECK-LABEL: func @test_gaussian_elimination_non_empty_set4() {
func @test_gaussian_elimination_non_empty_set4() {
  %c7 = constant 7 : index
  %c11 = constant 11 : index
  affine.for %arg0 = 1 to 10 {
    affine.for %arg1 = 1 to 100 {
      // CHECK: #[[$SET_7_11]](%arg0, %arg1)
      affine.if #set_2d_non_empty(%arg0, %arg1)[%c7, %c11] {
        call @external() : () -> ()
      }
    }
  }
  return
}

// Add invalid constraints to previous non-empty set to make it empty.
#set_2d_empty = affine_set<(d0, d1)[s0, s1] : (d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0,
                                       d0 * 5 - d1 * 11 + s0 * 7 + s1 == 0,
                                       d0 * 11 + d1 * 7 - s0 * 5 + s1 == 0,
                                       d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0,
                                       d0 - 1 == 0, d0 + 2 == 0)>

// CHECK-LABEL: func @test_gaussian_elimination_empty_set5() {
func @test_gaussian_elimination_empty_set5() {
  %c7 = constant 7 : index
  %c11 = constant 11 : index
  affine.for %arg0 = 1 to 10 {
    affine.for %arg1 = 1 to 100 {
      // CHECK: #[[$SET_EMPTY]]()
      affine.if #set_2d_empty(%arg0, %arg1)[%c7, %c11] {
        call @external() : () -> ()
      }
    }
  }
  return
}

// This is an artificially created system to exercise the worst case behavior of
// FM elimination - as a safeguard against improperly constructed constraint
// systems or fuzz input.
#set_fuzz_virus = affine_set<(d0, d1, d2, d3, d4, d5) : (
                            1089234*d0 + 203472*d1 + 82342 >= 0,
                            -55*d0 + 24*d1 + 238*d2 - 234*d3 - 9743 >= 0,
                            -5445*d0 - 284*d1 + 23*d2 + 34*d3 - 5943 >= 0,
                            -5445*d0 + 284*d1 + 238*d2 - 34*d3 >= 0,
                            445*d0 + 284*d1 + 238*d2 + 39*d3 >= 0,
                            -545*d0 + 214*d1 + 218*d2 - 94*d3 >= 0,
                            44*d0 - 184*d1 - 231*d2 + 14*d3 >= 0,
                            -45*d0 + 284*d1 + 138*d2 - 39*d3 >= 0,
                            154*d0 - 84*d1 + 238*d2 - 34*d3 >= 0,
                            54*d0 - 284*d1 - 223*d2 + 384*d3 >= 0,
                            -55*d0 + 284*d1 + 23*d2 + 34*d3 >= 0,
                            54*d0 - 84*d1 + 28*d2 - 34*d3 >= 0,
                            54*d0 - 24*d1 - 23*d2 + 34*d3 >= 0,
                            -55*d0 + 24*d1 + 23*d2 + 4*d3 >= 0,
                            15*d0 - 84*d1 + 238*d2 - 3*d3 >= 0,
                            5*d0 - 24*d1 - 223*d2 + 84*d3 >= 0,
                            -5*d0 + 284*d1 + 23*d2 - 4*d3 >= 0,
                            14*d0 + 4*d2 + 7234 >= 0,
                            -174*d0 - 534*d2 + 9834 >= 0,
                            194*d0 - 954*d2 + 9234 >= 0,
                            47*d0 - 534*d2 + 9734 >= 0,
                            -194*d0 - 934*d2 + 984 >= 0,
                            -947*d0 - 953*d2 + 234 >= 0,
                            184*d0 - 884*d2 + 884 >= 0,
                            -174*d0 + 834*d2 + 234 >= 0,
                            844*d0 + 634*d2 + 9874 >= 0,
                            -797*d2 - 79*d3 + 257 >= 0,
                            2039*d0 + 793*d2 - 99*d3 - 24*d4 + 234*d5 >= 0,
                            78*d2 - 788*d5 + 257 >= 0,
                            d3 - (d5 + 97*d0) floordiv 423 >= 0,
                            234* (d0 + d3 mod 5 floordiv 2342) mod 2309
                            + (d0 + 2038*d3) floordiv 208 >= 0,
                            239* (d0 + 2300 * d3) floordiv 2342
                            mod 2309 mod 239423 == 0,
                            d0 + d3 mod 2642 + (d3 + 2*d0) mod 1247
                            mod 2038 mod 2390 mod 2039 floordiv 55 >= 0
)>

// CHECK-LABEL: func @test_fuzz_explosion
func @test_fuzz_explosion(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index) {
  affine.for %arg4 = 1 to 10 {
    affine.for %arg5 = 1 to 100 {
      affine.if #set_fuzz_virus(%arg4, %arg5, %arg0, %arg1, %arg2, %arg3) {
        call @external() : () -> ()
      }
    }
  }
  return
}

// CHECK-LABEL: func @test_empty_set(%arg0: index) {
func @test_empty_set(%N : index) {
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 10 {
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0, d1) : (d0 - d1 >= 0, d1 - d0 - 1 >= 0)>(%i, %j) {
        "foo"() : () -> ()
      }
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0) : (d0 >= 0, -d0 - 1 >= 0)>(%i) {
        "bar"() : () -> ()
      }
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0) : (d0 >= 0, -d0 - 1 >= 0)>(%i) {
        "foo"() : () -> ()
      }
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0)[s0, s1] : (d0 >= 0, -d0 + s0 - 1 >= 0, -s0 >= 0)>(%i)[%N, %N] {
        "bar"() : () -> ()
      }
      // CHECK: affine.if #[[$SET_EMPTY]]()
      // The set below implies d0 = d1; so d1 >= d0, but d0 >= d1 + 1.
      affine.if affine_set<(d0, d1, d2) : (d0 - d1 == 0, d2 - d0 >= 0, d0 - d1 - 1 >= 0)>(%i, %j, %N) {
        "foo"() : () -> ()
      }
      // CHECK: affine.if #[[$SET_EMPTY]]()
      // The set below has rational solutions but no integer solutions; GCD test catches it.
      affine.if affine_set<(d0, d1) : (d0*2 -d1*2 - 1 == 0, d0 >= 0, -d0 + 100 >= 0, d1 >= 0, -d1 + 100 >= 0)>(%i, %j) {
        "foo"() : () -> ()
      }
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0, d1) : (d1 == 0, d0 - 1 >= 0, - d0 - 1 >= 0)>(%i, %j) {
        "foo"() : () -> ()
      }
    }
  }
  // The tests below test GCDTightenInequalities().
  affine.for %k = 0 to 10 {
    affine.for %l = 0 to 10 {
      // Empty because no multiple of 8 lies between 4 and 7.
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0) : (8*d0 - 4 >= 0, -8*d0 + 7 >= 0)>(%k) {
        "foo"() : () -> ()
      }
      // Same as above but with equalities and inequalities.
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0, d1) : (d0 - 4*d1 == 0, 4*d1 - 5 >= 0, -4*d1 + 7 >= 0)>(%k, %l) {
        "foo"() : () -> ()
      }
      // Same as above but with a combination of multiple identifiers. 4*d0 +
      // 8*d1 here is a multiple of 4, and so can't lie between 9 and 11. GCD
      // tightening will tighten constraints to 4*d0 + 8*d1 >= 12 and 4*d0 +
      // 8*d1 <= 8; hence infeasible.
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0, d1) : (4*d0 + 8*d1 - 9 >= 0, -4*d0 - 8*d1 + 11 >= 0)>(%k, %l) {
        "foo"() : () -> ()
      }
      // Same as above but with equalities added into the mix.
      // CHECK: affine.if #[[$SET_EMPTY]]()
      affine.if affine_set<(d0, d1, d2) : (d0 - 4*d2 == 0, d0 + 8*d1 - 9 >= 0, -d0 - 8*d1 + 11 >= 0)>(%k, %k, %l) {
        "foo"() : () -> ()
      }
    }
  }

  affine.for %m = 0 to 10 {
    // CHECK: affine.if #[[$SET_EMPTY]]()
    affine.if affine_set<(d0) : (d0 mod 2 - 3 == 0)> (%m) {
      "foo"() : () -> ()
    }
  }

  return
}

// -----

// An external function that we will use in bodies to avoid DCE.
func private @external() -> ()

// CHECK-DAG: #[[$SET:.*]] = affine_set<()[s0] : (s0 >= 0, -s0 + 50 >= 0)
// CHECK-DAG: #[[$EMPTY_SET:.*]] = affine_set<() : (1 == 0)
// CHECK-DAG: #[[$UNIV_SET:.*]] = affine_set<() : (0 == 0)

// CHECK-LABEL: func @simplify_set
func @simplify_set(%a : index, %b : index) {
  // CHECK: affine.if #[[$SET]]
  affine.if affine_set<(d0, d1) : (d0 - d1 + d1 + d0 >= 0, 2 >= 0, d0 >= 0, -d0 + 50 >= 0, -d0 + 100 >= 0)>(%a, %b) {
    call @external() : () -> ()
  }
  // CHECK: affine.if #[[$EMPTY_SET]]
  affine.if affine_set<(d0, d1) : (d0 mod 2 - 1 == 0, d0 - 2 * (d0 floordiv 2) == 0)>(%a, %b) {
    call @external() : () -> ()
  }
  // CHECK: affine.if #[[$UNIV_SET]]
  affine.if affine_set<(d0, d1) : (1 >= 0, 3 >= 0)>(%a, %b) {
    call @external() : () -> ()
  }
	return
}

// -----

// CHECK-DAG: -> (s0 * 2 + 1)

// Test "op local" simplification on affine.apply. DCE on addi will not happen.
func @affine.apply(%N : index) {
  %v = affine.apply affine_map<(d0, d1) -> (d0 + d1 + 1)>(%N, %N)
  addi %v, %v : index
  // CHECK: affine.apply #map{{.*}}()[%arg0]
  // CHECK-NEXT: addi
  return
}

// -----

// CHECK-LABEL: func @simplify_zero_dim_map
func @simplify_zero_dim_map(%in : memref<f32>) -> f32 {
  %out = affine.load %in[] : memref<f32>
  return %out : f32
}

// -----

// Tests the simplification of a semi-affine expression in various cases.
// CHECK-DAG: #[[$map0:.*]] = affine_map<()[s0, s1] -> (-(s1 floordiv s0) + 2)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<()[s0, s1] -> (-(s1 floordiv s0) + 42)>

// Tests the simplification of a semi-affine expression with a modulo operation on a floordiv and multiplication.
// CHECK-LABEL: func @semiaffine_mod
func @semiaffine_mod(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] ->((-((d0 floordiv s0) * s0) + s0 * s0) mod s0)> (%arg0)[%arg1]
  // CHECK:       %[[CST:.*]] = constant 0
  return %a : index
}

// Tests the simplification of a semi-affine expression with a nested floordiv and a floordiv on modulo operation.
// CHECK-LABEL: func @semiaffine_floordiv
func @semiaffine_floordiv(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] ->((-((d0 floordiv s0) * s0) + ((2 * s0) mod (3 * s0))) floordiv s0)> (%arg0)[%arg1]
  // CHECK: affine.apply #[[$map0]]()[%arg1, %arg0]
  return %a : index
}

// Tests the simplification of a semi-affine expression with a ceildiv operation and a division of constant 0 by a symbol.
// CHECK-LABEL: func @semiaffine_ceildiv
func @semiaffine_ceildiv(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] ->((-((d0 floordiv s0) * s0) + s0 * 42 + ((5-5) floordiv s0)) ceildiv  s0)> (%arg0)[%arg1]
  // CHECK: affine.apply #[[$map1]]()[%arg1, %arg0]
  return %a : index
}

// Tests the simplification of a semi-affine expression with a nested ceildiv operation and further simplifications after performing ceildiv.
// CHECK-LABEL: func @semiaffine_composite_floor
func @semiaffine_composite_floor(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] ->(((((s0 * 2) ceildiv 4) * 5) + s0 * 42) ceildiv s0)> (%arg0)[%arg1]
  // CHECK:       %[[CST:.*]] = constant 47
  return %a : index
}

// Tests the simplification of a semi-affine expression with a modulo operation with a second operand that simplifies to symbol.
// CHECK-LABEL: func @semiaffine_unsimplified_symbol
func @semiaffine_unsimplified_symbol(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] ->(s0 mod (2 * s0 - s0))> (%arg0)[%arg1]
  // CHECK:       %[[CST:.*]] = constant 0
  return %a : index
}

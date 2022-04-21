// RUN: mlir-opt -lower-affine %s | FileCheck %s

// CHECK-LABEL: func @empty() {
func.func @empty() {
  return     // CHECK:  return
}            // CHECK: }

func.func private @body(index) -> ()

// Simple loops are properly converted.
// CHECK-LABEL: func @simple_loop
// CHECK-NEXT:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[c42:.*]] = arith.constant 42 : index
// CHECK-NEXT:   %[[c1_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c1]] to %[[c42]] step %[[c1_0]] {
// CHECK-NEXT:     call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @simple_loop() {
  affine.for %i = 1 to 42 {
    func.call @body(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

func.func @for_with_yield(%buffer: memref<1024xf32>) -> (f32) {
  %sum_0 = arith.constant 0.0 : f32
  %sum = affine.for %i = 0 to 10 step 2 iter_args(%sum_iter = %sum_0) -> (f32) {
    %t = affine.load %buffer[%i] : memref<1024xf32>
    %sum_next = arith.addf %sum_iter, %t : f32
    affine.yield %sum_next : f32
  }
  return %sum : f32
}

// CHECK-LABEL: func @for_with_yield
// CHECK:         %[[INIT_SUM:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[LOWER:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[UPPER:.*]] = arith.constant 10 : index
// CHECK-NEXT:    %[[STEP:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[SUM:.*]] = scf.for %[[IV:.*]] = %[[LOWER]] to %[[UPPER]] step %[[STEP]] iter_args(%[[SUM_ITER:.*]] = %[[INIT_SUM]]) -> (f32) {
// CHECK-NEXT:      memref.load
// CHECK-NEXT:      %[[SUM_NEXT:.*]] = arith.addf
// CHECK-NEXT:      scf.yield %[[SUM_NEXT]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[SUM]] : f32

/////////////////////////////////////////////////////////////////////

func.func private @pre(index) -> ()
func.func private @body2(index, index) -> ()
func.func private @post(index) -> ()

// CHECK-LABEL: func @imperfectly_nested_loops
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[c42:.*]] = arith.constant 42 : index
// CHECK-NEXT:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[c42]] step %[[c1]] {
// CHECK-NEXT:     call @pre(%{{.*}}) : (index) -> ()
// CHECK-NEXT:     %[[c7:.*]] = arith.constant 7 : index
// CHECK-NEXT:     %[[c56:.*]] = arith.constant 56 : index
// CHECK-NEXT:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     for %{{.*}} = %[[c7]] to %[[c56]] step %[[c2]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     call @post(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @imperfectly_nested_loops() {
  affine.for %i = 0 to 42 {
    func.call @pre(%i) : (index) -> ()
    affine.for %j = 7 to 56 step 2 {
      func.call @body2(%i, %j) : (index, index) -> ()
    }
    func.call @post(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

func.func private @mid(index) -> ()
func.func private @body3(index, index) -> ()

// CHECK-LABEL: func @more_imperfectly_nested_loops
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[c42:.*]] = arith.constant 42 : index
// CHECK-NEXT:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[c42]] step %[[c1]] {
// CHECK-NEXT:     call @pre(%{{.*}}) : (index) -> ()
// CHECK-NEXT:     %[[c7:.*]] = arith.constant 7 : index
// CHECK-NEXT:     %[[c56:.*]] = arith.constant 56 : index
// CHECK-NEXT:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     for %{{.*}} = %[[c7]] to %[[c56]] step %[[c2]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     call @mid(%{{.*}}) : (index) -> ()
// CHECK-NEXT:     %[[c18:.*]] = arith.constant 18 : index
// CHECK-NEXT:     %[[c37:.*]] = arith.constant 37 : index
// CHECK-NEXT:     %[[c3:.*]] = arith.constant 3 : index
// CHECK-NEXT:     for %{{.*}} = %[[c18]] to %[[c37]] step %[[c3]] {
// CHECK-NEXT:       call @body3(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     call @post(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @more_imperfectly_nested_loops() {
  affine.for %i = 0 to 42 {
    func.call @pre(%i) : (index) -> ()
    affine.for %j = 7 to 56 step 2 {
      func.call @body2(%i, %j) : (index, index) -> ()
    }
    func.call @mid(%i) : (index) -> ()
    affine.for %k = 18 to 37 step 3 {
      func.call @body3(%i, %k) : (index, index) -> ()
    }
    func.call @post(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @affine_apply_loops_shorthand
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %{{.*}} step %[[c1]] {
// CHECK-NEXT:     %[[c42:.*]] = arith.constant 42 : index
// CHECK-NEXT:     %[[c1_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     for %{{.*}} = %{{.*}} to %[[c42]] step %[[c1_0]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @affine_apply_loops_shorthand(%N : index) {
  affine.for %i = 0 to %N {
    affine.for %j = affine_map<(d0)[]->(d0)>(%i)[] to 42 {
      func.call @body2(%i, %j) : (index, index) -> ()
    }
  }
  return
}

/////////////////////////////////////////////////////////////////////

func.func private @get_idx() -> (index)

#set1 = affine_set<(d0) : (20 - d0 >= 0)>
#set2 = affine_set<(d0) : (d0 - 10 >= 0)>

// CHECK-LABEL: func @if_only
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = arith.constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = arith.muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = arith.constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = arith.addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = arith.cmpi sge, %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @if_only() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    func.call @body(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @if_else
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = arith.constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = arith.muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = arith.constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = arith.addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = arith.cmpi sge, %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   } else {
// CHECK-NEXT:     call @mid(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @if_else() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    func.call @body(%i) : (index) -> ()
  } else {
    func.call @mid(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @nested_ifs
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = arith.constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = arith.muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = arith.constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = arith.addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = arith.cmpi sge, %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     %[[c0_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[cm10:.*]] = arith.constant -10 : index
// CHECK-NEXT:     %[[v4:.*]] = arith.addi %[[v0]], %[[cm10]] : index
// CHECK-NEXT:     %[[v5:.*]] = arith.cmpi sge, %[[v4]], %[[c0_0]] : index
// CHECK-NEXT:     if %[[v5]] {
// CHECK-NEXT:       call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %[[c0_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[cm10:.*]] = arith.constant -10 : index
// CHECK-NEXT:     %{{.*}} = arith.addi %[[v0]], %[[cm10]] : index
// CHECK-NEXT:     %{{.*}} = arith.cmpi sge, %{{.*}}, %[[c0_0]] : index
// CHECK-NEXT:     if %{{.*}} {
// CHECK-NEXT:       call @mid(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @nested_ifs() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    affine.if #set2(%i) {
      func.call @body(%i) : (index) -> ()
    }
  } else {
    affine.if #set2(%i) {
      func.call @mid(%i) : (index) -> ()
    }
  }
  return
}

// CHECK-LABEL: func @if_with_yield
// CHECK-NEXT:   %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-NEXT:   %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[cm10:.*]] = arith.constant -10 : index
// CHECK-NEXT:   %[[v1:.*]] = arith.addi %[[v0]], %[[cm10]] : index
// CHECK-NEXT:   %[[v2:.*]] = arith.cmpi sge, %[[v1]], %[[c0]] : index
// CHECK-NEXT:   %[[v3:.*]] = scf.if %[[v2]] -> (i64) {
// CHECK-NEXT:     scf.yield %[[c0_i64]] : i64
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %[[c1_i64]] : i64
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[v3]] : i64
// CHECK-NEXT: }
func.func @if_with_yield() -> (i64) {
  %cst0 = arith.constant 0 : i64
  %cst1 = arith.constant 1 : i64
  %i = call @get_idx() : () -> (index)
  %1 = affine.if #set2(%i) -> (i64) {
      affine.yield %cst0 : i64
  } else {
      affine.yield %cst1 : i64
  }
  return %1 : i64
}

#setN = affine_set<(d0)[N,M,K,L] : (N - d0 + 1 >= 0, N - 1 >= 0, M - 1 >= 0, K - 1 >= 0, L - 42 == 0)>

// CHECK-LABEL: func @multi_cond
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = arith.constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = arith.muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[v2:.*]] = arith.addi %[[v1]], %{{.*}} : index
// CHECK-NEXT:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[v3:.*]] = arith.addi %[[v2]], %[[c1]] : index
// CHECK-NEXT:   %[[v4:.*]] = arith.cmpi sge, %[[v3]], %[[c0]] : index
// CHECK-NEXT:   %[[cm1_0:.*]] = arith.constant -1 : index
// CHECK-NEXT:   %[[v5:.*]] = arith.addi %{{.*}}, %[[cm1_0]] : index
// CHECK-NEXT:   %[[v6:.*]] = arith.cmpi sge, %[[v5]], %[[c0]] : index
// CHECK-NEXT:   %[[v7:.*]] = arith.andi %[[v4]], %[[v6]] : i1
// CHECK-NEXT:   %[[cm1_1:.*]] = arith.constant -1 : index
// CHECK-NEXT:   %[[v8:.*]] = arith.addi %{{.*}}, %[[cm1_1]] : index
// CHECK-NEXT:   %[[v9:.*]] = arith.cmpi sge, %[[v8]], %[[c0]] : index
// CHECK-NEXT:   %[[v10:.*]] = arith.andi %[[v7]], %[[v9]] : i1
// CHECK-NEXT:   %[[cm1_2:.*]] = arith.constant -1 : index
// CHECK-NEXT:   %[[v11:.*]] = arith.addi %{{.*}}, %[[cm1_2]] : index
// CHECK-NEXT:   %[[v12:.*]] = arith.cmpi sge, %[[v11]], %[[c0]] : index
// CHECK-NEXT:   %[[v13:.*]] = arith.andi %[[v10]], %[[v12]] : i1
// CHECK-NEXT:   %[[cm42:.*]] = arith.constant -42 : index
// CHECK-NEXT:   %[[v14:.*]] = arith.addi %{{.*}}, %[[cm42]] : index
// CHECK-NEXT:   %[[v15:.*]] = arith.cmpi eq, %[[v14]], %[[c0]] : index
// CHECK-NEXT:   %[[v16:.*]] = arith.andi %[[v13]], %[[v15]] : i1
// CHECK-NEXT:   if %[[v16]] {
// CHECK-NEXT:     call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   } else {
// CHECK-NEXT:     call @mid(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @multi_cond(%N : index, %M : index, %K : index, %L : index) {
  %i = call @get_idx() : () -> (index)
  affine.if #setN(%i)[%N,%M,%K,%L] {
    func.call @body(%i) : (index) -> ()
  } else {
    func.call @mid(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @if_for
func.func @if_for() {
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
  %i = call @get_idx() : () -> (index)
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = arith.constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = arith.muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = arith.constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = arith.addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = arith.cmpi sge, %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     %[[c0:.*]]{{.*}} = arith.constant 0 : index
// CHECK-NEXT:     %[[c42:.*]]{{.*}} = arith.constant 42 : index
// CHECK-NEXT:     %[[c1:.*]]{{.*}} = arith.constant 1 : index
// CHECK-NEXT:     for %{{.*}} = %[[c0:.*]]{{.*}} to %[[c42:.*]]{{.*}} step %[[c1:.*]]{{.*}} {
// CHECK-NEXT:       %[[c0_:.*]]{{.*}} = arith.constant 0 : index
// CHECK-NEXT:       %[[cm10:.*]] = arith.constant -10 : index
// CHECK-NEXT:       %[[v4:.*]] = arith.addi %{{.*}}, %[[cm10]] : index
// CHECK-NEXT:       %[[v5:.*]] = arith.cmpi sge, %[[v4]], %[[c0_:.*]]{{.*}} : index
// CHECK-NEXT:       if %[[v5]] {
// CHECK-NEXT:         call @body2(%[[v0]], %{{.*}}) : (index, index) -> ()
  affine.if #set1(%i) {
    affine.for %j = 0 to 42 {
      affine.if #set2(%j) {
        func.call @body2(%i, %j) : (index, index) -> ()
      }
    }
  }
//      CHECK:   %[[c0:.*]]{{.*}} = arith.constant 0 : index
// CHECK-NEXT:   %[[c42:.*]]{{.*}} = arith.constant 42 : index
// CHECK-NEXT:   %[[c1:.*]]{{.*}} = arith.constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0:.*]]{{.*}} to %[[c42:.*]]{{.*}} step %[[c1:.*]]{{.*}} {
// CHECK-NEXT:     %[[c0:.*]]{{.*}} = arith.constant 0 : index
// CHECK-NEXT:     %[[cm10:.*]]{{.*}} = arith.constant -10 : index
// CHECK-NEXT:     %{{.*}} = arith.addi %{{.*}}, %[[cm10:.*]]{{.*}} : index
// CHECK-NEXT:     %{{.*}} = arith.cmpi sge, %{{.*}}, %[[c0:.*]]{{.*}} : index
// CHECK-NEXT:     if %{{.*}} {
// CHECK-NEXT:       %[[c0_:.*]]{{.*}} = arith.constant 0 : index
// CHECK-NEXT:       %[[c42_:.*]]{{.*}} = arith.constant 42 : index
// CHECK-NEXT:       %[[c1_:.*]]{{.*}} = arith.constant 1 : index
// CHECK-NEXT:       for %{{.*}} = %[[c0_:.*]]{{.*}} to %[[c42_:.*]]{{.*}} step %[[c1_:.*]]{{.*}} {
  affine.for %k = 0 to 42 {
    affine.if #set2(%k) {
      affine.for %l = 0 to 42 {
        func.call @body3(%k, %l) : (index, index) -> ()
      }
    }
  }
//      CHECK:   return
  return
}

#lbMultiMap = affine_map<(d0)[s0] -> (d0, s0 - d0)>
#ubMultiMap = affine_map<(d0)[s0] -> (s0, d0 + 10)>

// CHECK-LABEL: func @loop_min_max
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[c42:.*]] = arith.constant 42 : index
// CHECK-NEXT:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[c42]] step %[[c1]] {
// CHECK-NEXT:     %[[cm1:.*]] = arith.constant -1 : index
// CHECK-NEXT:     %[[a:.*]] = arith.muli %{{.*}}, %[[cm1]] : index
// CHECK-NEXT:     %[[b:.*]] = arith.addi %[[a]], %{{.*}} : index
// CHECK-NEXT:     %[[c:.*]] = arith.cmpi sgt, %{{.*}}, %[[b]] : index
// CHECK-NEXT:     %[[d:.*]] = arith.select %[[c]], %{{.*}}, %[[b]] : index
// CHECK-NEXT:     %[[c10:.*]] = arith.constant 10 : index
// CHECK-NEXT:     %[[e:.*]] = arith.addi %{{.*}}, %[[c10]] : index
// CHECK-NEXT:     %[[f:.*]] = arith.cmpi slt, %{{.*}}, %[[e]] : index
// CHECK-NEXT:     %[[g:.*]] = arith.select %[[f]], %{{.*}}, %[[e]] : index
// CHECK-NEXT:     %[[c1_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     for %{{.*}} = %[[d]] to %[[g]] step %[[c1_0]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @loop_min_max(%N : index) {
  affine.for %i = 0 to 42 {
    affine.for %j = max #lbMultiMap(%i)[%N] to min #ubMultiMap(%i)[%N] {
      func.call @body2(%i, %j) : (index, index) -> ()
    }
  }
  return
}

#map_7_values = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>

// Check that the "min" (cmpi slt + select) reduction sequence is emitted
// correctly for an affine map with 7 results.

// CHECK-LABEL: func @min_reduction_tree
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[c01:.+]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %[[r01:.+]] = arith.select %[[c01]], %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %[[c012:.+]] = arith.cmpi slt, %[[r01]], %{{.*}} : index
// CHECK-NEXT:   %[[r012:.+]] = arith.select %[[c012]], %[[r01]], %{{.*}} : index
// CHECK-NEXT:   %[[c0123:.+]] = arith.cmpi slt, %[[r012]], %{{.*}} : index
// CHECK-NEXT:   %[[r0123:.+]] = arith.select %[[c0123]], %[[r012]], %{{.*}} : index
// CHECK-NEXT:   %[[c01234:.+]] = arith.cmpi slt, %[[r0123]], %{{.*}} : index
// CHECK-NEXT:   %[[r01234:.+]] = arith.select %[[c01234]], %[[r0123]], %{{.*}} : index
// CHECK-NEXT:   %[[c012345:.+]] = arith.cmpi slt, %[[r01234]], %{{.*}} : index
// CHECK-NEXT:   %[[r012345:.+]] = arith.select %[[c012345]], %[[r01234]], %{{.*}} : index
// CHECK-NEXT:   %[[c0123456:.+]] = arith.cmpi slt, %[[r012345]], %{{.*}} : index
// CHECK-NEXT:   %[[r0123456:.+]] = arith.select %[[c0123456]], %[[r012345]], %{{.*}} : index
// CHECK-NEXT:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[r0123456]] step %[[c1]] {
// CHECK-NEXT:     call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @min_reduction_tree(%v1 : index, %v2 : index, %v3 : index, %v4 : index, %v5 : index, %v6 : index, %v7 : index) {
  affine.for %i = 0 to min #map_7_values(%v1, %v2, %v3, %v4, %v5, %v6, %v7)[] {
    func.call @body(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0 + 1)>
#map4 = affine_map<(d0,d1,d2,d3)[s0,s1,s2] -> (d0 + 2*d1 + 3*d2 + 4*d3 + 5*s0 + 6*s1 + 7*s2)>
#map5 = affine_map<(d0,d1,d2) -> (d0,d1,d2)>
#map6 = affine_map<(d0,d1,d2) -> (d0 + d1 + d2)>

// CHECK-LABEL: func @affine_applies(
func.func @affine_applies(%arg0 : index) {
// CHECK: %[[c0:.*]] = arith.constant 0 : index
  %zero = affine.apply #map0()

// Identity maps are just discarded.
// CHECK-NEXT: %[[c101:.*]] = arith.constant 101 : index
  %101 = arith.constant 101 : index
  %symbZero = affine.apply #map1()[%zero]
// CHECK-NEXT: %[[c102:.*]] = arith.constant 102 : index
  %102 = arith.constant 102 : index
  %copy = affine.apply #map2(%zero)

// CHECK-NEXT: %[[v0:.*]] = arith.addi %[[c0]], %[[c0]] : index
// CHECK-NEXT: %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[v1:.*]] = arith.addi %[[v0]], %[[c1]] : index
  %one = affine.apply #map3(%symbZero)[%zero]

// CHECK-NEXT: %[[c2:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[v2:.*]] = arith.muli %arg0, %[[c2]] : index
// CHECK-NEXT: %[[v3:.*]] = arith.addi %arg0, %[[v2]] : index
// CHECK-NEXT: %[[c3:.*]] = arith.constant 3 : index
// CHECK-NEXT: %[[v4:.*]] = arith.muli %arg0, %[[c3]] : index
// CHECK-NEXT: %[[v5:.*]] = arith.addi %[[v3]], %[[v4]] : index
// CHECK-NEXT: %[[c4:.*]] = arith.constant 4 : index
// CHECK-NEXT: %[[v6:.*]] = arith.muli %arg0, %[[c4]] : index
// CHECK-NEXT: %[[v7:.*]] = arith.addi %[[v5]], %[[v6]] : index
// CHECK-NEXT: %[[c5:.*]] = arith.constant 5 : index
// CHECK-NEXT: %[[v8:.*]] = arith.muli %arg0, %[[c5]] : index
// CHECK-NEXT: %[[v9:.*]] = arith.addi %[[v7]], %[[v8]] : index
// CHECK-NEXT: %[[c6:.*]] = arith.constant 6 : index
// CHECK-NEXT: %[[v10:.*]] = arith.muli %arg0, %[[c6]] : index
// CHECK-NEXT: %[[v11:.*]] = arith.addi %[[v9]], %[[v10]] : index
// CHECK-NEXT: %[[c7:.*]] = arith.constant 7 : index
// CHECK-NEXT: %[[v12:.*]] = arith.muli %arg0, %[[c7]] : index
// CHECK-NEXT: %[[v13:.*]] = arith.addi %[[v11]], %[[v12]] : index
  %four = affine.apply #map4(%arg0, %arg0, %arg0, %arg0)[%arg0, %arg0, %arg0]
  return
}

// CHECK-LABEL: func @args_ret_affine_apply(
func.func @args_ret_affine_apply(index, index) -> (index, index) {
^bb0(%0 : index, %1 : index):
// CHECK-NEXT: return %{{.*}}, %{{.*}} : index, index
  %00 = affine.apply #map2 (%0)
  %11 = affine.apply #map1 ()[%1]
  return %00, %11 : index, index
}

//===---------------------------------------------------------------------===//
// Test lowering of Euclidean (floor) division, ceil division and modulo
// operation used in affine expressions.  In addition to testing the
// operation-level output, check that the obtained results are correct by
// applying constant folding transformation after affine lowering.
//===---------------------------------------------------------------------===//

#mapmod = affine_map<(i) -> (i mod 42)>

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_mod
func.func @affine_apply_mod(%arg0 : index) -> (index) {
// CHECK-NEXT: %[[c42:.*]] = arith.constant 42 : index
// CHECK-NEXT: %[[v0:.*]] = arith.remsi %{{.*}}, %[[c42]] : index
// CHECK-NEXT: %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[v1:.*]] = arith.cmpi slt, %[[v0]], %[[c0]] : index
// CHECK-NEXT: %[[v2:.*]] = arith.addi %[[v0]], %[[c42]] : index
// CHECK-NEXT: %[[v3:.*]] = arith.select %[[v1]], %[[v2]], %[[v0]] : index
  %0 = affine.apply #mapmod (%arg0)
  return %0 : index
}

#mapfloordiv = affine_map<(i) -> (i floordiv 42)>

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_floordiv
func.func @affine_apply_floordiv(%arg0 : index) -> (index) {
// CHECK-NEXT: %[[c42:.*]] = arith.constant 42 : index
// CHECK-NEXT: %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[cm1:.*]] = arith.constant -1 : index
// CHECK-NEXT: %[[v0:.*]] = arith.cmpi slt, %{{.*}}, %[[c0]] : index
// CHECK-NEXT: %[[v1:.*]] = arith.subi %[[cm1]], %{{.*}} : index
// CHECK-NEXT: %[[v2:.*]] = arith.select %[[v0]], %[[v1]], %{{.*}} : index
// CHECK-NEXT: %[[v3:.*]] = arith.divsi %[[v2]], %[[c42]] : index
// CHECK-NEXT: %[[v4:.*]] = arith.subi %[[cm1]], %[[v3]] : index
// CHECK-NEXT: %[[v5:.*]] = arith.select %[[v0]], %[[v4]], %[[v3]] : index
  %0 = affine.apply #mapfloordiv (%arg0)
  return %0 : index
}

#mapceildiv = affine_map<(i) -> (i ceildiv 42)>

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_ceildiv
func.func @affine_apply_ceildiv(%arg0 : index) -> (index) {
// CHECK-NEXT:  %[[c42:.*]] = arith.constant 42 : index
// CHECK-NEXT:  %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:  %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:  %[[v0:.*]] = arith.cmpi sle, %{{.*}}, %[[c0]] : index
// CHECK-NEXT:  %[[v1:.*]] = arith.subi %[[c0]], %{{.*}} : index
// CHECK-NEXT:  %[[v2:.*]] = arith.subi %{{.*}}, %[[c1]] : index
// CHECK-NEXT:  %[[v3:.*]] = arith.select %[[v0]], %[[v1]], %[[v2]] : index
// CHECK-NEXT:  %[[v4:.*]] = arith.divsi %[[v3]], %[[c42]] : index
// CHECK-NEXT:  %[[v5:.*]] = arith.subi %[[c0]], %[[v4]] : index
// CHECK-NEXT:  %[[v6:.*]] = arith.addi %[[v4]], %[[c1]] : index
// CHECK-NEXT:  %[[v7:.*]] = arith.select %[[v0]], %[[v5]], %[[v6]] : index
  %0 = affine.apply #mapceildiv (%arg0)
  return %0 : index
}

// CHECK-LABEL: func @affine_load
func.func @affine_load(%arg0 : index) {
  %0 = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    %1 = affine.load %0[%i0 + symbol(%arg0) + 7] : memref<10xf32>
  }
// CHECK:       %[[a:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %[[c7:.*]] = arith.constant 7 : index
// CHECK-NEXT:  %[[b:.*]] = arith.addi %[[a]], %[[c7]] : index
// CHECK-NEXT:  %{{.*}} = memref.load %[[v0:.*]][%[[b]]] : memref<10xf32>
  return
}

// CHECK-LABEL: func @affine_store
func.func @affine_store(%arg0 : index) {
  %0 = memref.alloc() : memref<10xf32>
  %1 = arith.constant 11.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %1, %0[%i0 - symbol(%arg0) + 7] : memref<10xf32>
  }
// CHECK:       %c-1 = arith.constant -1 : index
// CHECK-NEXT:  %[[a:.*]] = arith.muli %arg0, %c-1 : index
// CHECK-NEXT:  %[[b:.*]] = arith.addi %{{.*}}, %[[a]] : index
// CHECK-NEXT:  %c7 = arith.constant 7 : index
// CHECK-NEXT:  %[[c:.*]] = arith.addi %[[b]], %c7 : index
// CHECK-NEXT:  store %cst, %0[%[[c]]] : memref<10xf32>
  return
}

// CHECK-LABEL: func @affine_load_store_zero_dim
func.func @affine_load_store_zero_dim(%arg0 : memref<i32>, %arg1 : memref<i32>) {
  %0 = affine.load %arg0[] : memref<i32>
  affine.store %0, %arg1[] : memref<i32>
// CHECK: %[[x:.*]] = memref.load %arg0[] : memref<i32>
// CHECK: store %[[x]], %arg1[] : memref<i32>
  return
}

// CHECK-LABEL: func @affine_prefetch
func.func @affine_prefetch(%arg0 : index) {
  %0 = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.prefetch %0[%i0 + symbol(%arg0) + 7], read, locality<3>, data : memref<10xf32>
  }
// CHECK:       %[[a:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %[[c7:.*]] = arith.constant 7 : index
// CHECK-NEXT:  %[[b:.*]] = arith.addi %[[a]], %[[c7]] : index
// CHECK-NEXT:  memref.prefetch %[[v0:.*]][%[[b]]], read, locality<3>, data : memref<10xf32>
  return
}

// CHECK-LABEL: func @affine_dma_start
func.func @affine_dma_start(%arg0 : index) {
  %0 = memref.alloc() : memref<100xf32>
  %1 = memref.alloc() : memref<100xf32, 2>
  %2 = memref.alloc() : memref<1xi32>
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.dma_start %0[%i0 + 7], %1[%arg0 + 11], %2[%c0], %c64
        : memref<100xf32>, memref<100xf32, 2>, memref<1xi32>
  }
// CHECK:       %c7 = arith.constant 7 : index
// CHECK-NEXT:  %[[a:.*]] = arith.addi %{{.*}}, %c7 : index
// CHECK-NEXT:  %c11 = arith.constant 11 : index
// CHECK-NEXT:  %[[b:.*]] = arith.addi %arg0, %c11 : index
// CHECK-NEXT:  dma_start %0[%[[a]]], %1[%[[b]]], %c64, %2[%c0] : memref<100xf32>, memref<100xf32, 2>, memref<1xi32>
  return
}

// CHECK-LABEL: func @affine_dma_wait
func.func @affine_dma_wait(%arg0 : index) {
  %2 = memref.alloc() : memref<1xi32>
  %c64 = arith.constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.dma_wait %2[%i0 + %arg0 + 17], %c64 : memref<1xi32>
  }
// CHECK:       %[[a:.*]] = arith.addi %{{.*}}, %arg0 : index
// CHECK-NEXT:  %c17 = arith.constant 17 : index
// CHECK-NEXT:  %[[b:.*]] = arith.addi %[[a]], %c17 : index
// CHECK-NEXT:  dma_wait %0[%[[b]]], %c64 : memref<1xi32>
  return
}

// CHECK-LABEL: func @affine_min
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @affine_min(%arg0: index, %arg1: index) -> index{
  // CHECK: %[[Cm1:.*]] = arith.constant -1
  // CHECK: %[[neg1:.*]] = arith.muli %[[ARG1]], %[[Cm1:.*]]
  // CHECK: %[[first:.*]] = arith.addi %[[ARG0]], %[[neg1]]
  // CHECK: %[[Cm2:.*]] = arith.constant -1
  // CHECK: %[[neg2:.*]] = arith.muli %[[ARG0]], %[[Cm2:.*]]
  // CHECK: %[[second:.*]] = arith.addi %[[ARG1]], %[[neg2]]
  // CHECK: %[[cmp:.*]] = arith.cmpi slt, %[[first]], %[[second]]
  // CHECK: arith.select %[[cmp]], %[[first]], %[[second]]
  %0 = affine.min affine_map<(d0,d1) -> (d0 - d1, d1 - d0)>(%arg0, %arg1)
  return %0 : index
}

// CHECK-LABEL: func @affine_max
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @affine_max(%arg0: index, %arg1: index) -> index{
  // CHECK: %[[Cm1:.*]] = arith.constant -1
  // CHECK: %[[neg1:.*]] = arith.muli %[[ARG1]], %[[Cm1:.*]]
  // CHECK: %[[first:.*]] = arith.addi %[[ARG0]], %[[neg1]]
  // CHECK: %[[Cm2:.*]] = arith.constant -1
  // CHECK: %[[neg2:.*]] = arith.muli %[[ARG0]], %[[Cm2:.*]]
  // CHECK: %[[second:.*]] = arith.addi %[[ARG1]], %[[neg2]]
  // CHECK: %[[cmp:.*]] = arith.cmpi sgt, %[[first]], %[[second]]
  // CHECK: arith.select %[[cmp]], %[[first]], %[[second]]
  %0 = affine.max affine_map<(d0,d1) -> (d0 - d1, d1 - d0)>(%arg0, %arg1)
  return %0 : index
}

// CHECK-LABEL: func @affine_parallel(
// CHECK-SAME: %[[ARG0:.*]]: memref<100x100xf32>, %[[ARG1:.*]]: memref<100x100xf32>) {
func.func @affine_parallel(%o: memref<100x100xf32>, %a: memref<100x100xf32>) {
  affine.parallel (%i, %j) = (0, 0) to (100, 100) {
  }
  return
}

// CHECK-DAG:    %[[C100:.*]] = arith.constant 100
// CHECK-DAG:    %[[C100_1:.*]] = arith.constant 100
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0
// CHECK-DAG:    %[[C0_1:.*]] = arith.constant 0
// CHECK-DAG:    %[[C1:.*]] = arith.constant 1
// CHECK-DAG:    %[[C1_1:.*]] = arith.constant 1
// CHECK-DAG:    scf.parallel (%arg2, %arg3) = (%[[C0]], %[[C0_1]]) to (%[[C100]], %[[C100_1]]) step (%[[C1]], %[[C1_1]]) {

// CHECK-LABEL: func @affine_parallel_tiled(
// CHECK-SAME: %[[ARG0:.*]]: memref<100x100xf32>, %[[ARG1:.*]]: memref<100x100xf32>, %[[ARG2:.*]]: memref<100x100xf32>) {
func.func @affine_parallel_tiled(%o: memref<100x100xf32>, %a: memref<100x100xf32>, %b: memref<100x100xf32>) {
  affine.parallel (%i0, %j0, %k0) = (0, 0, 0) to (100, 100, 100) step (10, 10, 10) {
    affine.parallel (%i1, %j1, %k1) = (%i0, %j0, %k0) to (%i0 + 10, %j0 + 10, %k0 + 10) {
      %0 = affine.load %a[%i1, %k1] : memref<100x100xf32>
      %1 = affine.load %b[%k1, %j1] : memref<100x100xf32>
      %2 = arith.mulf %0, %1 : f32
    }
  }
  return
}

// CHECK-DAG:     %[[C100:.*]] = arith.constant 100
// CHECK-DAG:     %[[C100_0:.*]] = arith.constant 100
// CHECK-DAG:     %[[C100_1:.*]] = arith.constant 100
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0
// CHECK-DAG:     %[[C0_2:.*]] = arith.constant 0
// CHECK-DAG:     %[[C0_3:.*]] = arith.constant 0
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10
// CHECK-DAG:     %[[C10_4:.*]] = arith.constant 10
// CHECK-DAG:     %[[C10_5:.*]] = arith.constant 10
// CHECK:         scf.parallel (%[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]]) = (%[[C0]], %[[C0_2]], %[[C0_3]]) to (%[[C100]], %[[C100_0]], %[[C100_1]]) step (%[[C10]], %[[C10_4]], %[[C10_5]]) {
// CHECK-DAG:       %[[C10_6:.*]] = arith.constant 10
// CHECK-DAG:       %[[A0:.*]] = arith.addi %[[arg3]], %[[C10_6]]
// CHECK-DAG:       %[[C10_7:.*]] = arith.constant 10
// CHECK-DAG:       %[[A1:.*]] = arith.addi %[[arg4]], %[[C10_7]]
// CHECK-DAG:       %[[C10_8:.*]] = arith.constant 10
// CHECK-DAG:       %[[A2:.*]] = arith.addi %[[arg5]], %[[C10_8]]
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1
// CHECK-DAG:       %[[C1_9:.*]] = arith.constant 1
// CHECK-DAG:       %[[C1_10:.*]] = arith.constant 1
// CHECK:           scf.parallel (%[[arg6:.*]], %[[arg7:.*]], %[[arg8:.*]]) = (%[[arg3]], %[[arg4]], %[[arg5]]) to (%[[A0]], %[[A1]], %[[A2]]) step (%[[C1]], %[[C1_9]], %[[C1_10]]) {
// CHECK:             %[[A3:.*]] = memref.load %[[ARG1]][%[[arg6]], %[[arg8]]] : memref<100x100xf32>
// CHECK:             %[[A4:.*]] = memref.load %[[ARG2]][%[[arg8]], %[[arg7]]] : memref<100x100xf32>
// CHECK:             arith.mulf %[[A3]], %[[A4]] : f32
// CHECK:             scf.yield

/////////////////////////////////////////////////////////////////////

func.func @affine_parallel_simple(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> (memref<3x3xf32>) {
  %O = memref.alloc() : memref<3x3xf32>
  affine.parallel (%kx, %ky) = (0, 0) to (2, 2) {
      %1 = affine.load %arg0[%kx, %ky] : memref<3x3xf32>
      %2 = affine.load %arg1[%kx, %ky] : memref<3x3xf32>
      %3 = arith.mulf %1, %2 : f32
      affine.store %3, %O[%kx, %ky] : memref<3x3xf32>
  }
  return %O : memref<3x3xf32>
}
// CHECK-LABEL: func @affine_parallel_simple
// CHECK:         %[[LOWER_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[UPPER_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[LOWER_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[UPPER_2:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[STEP_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[STEP_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:    scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[LOWER_1]], %[[LOWER_2]]) to (%[[UPPER_1]], %[[UPPER_2]]) step (%[[STEP_1]], %[[STEP_2]]) {
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.load
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.load
// CHECK-NEXT:      %[[PRODUCT:.*]] = arith.mulf
// CHECK-NEXT:      store
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////

func.func @affine_parallel_simple_dynamic_bounds(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %c_0 = arith.constant 0 : index
  %output_dim = memref.dim %arg0, %c_0 : memref<?x?xf32>
  affine.parallel (%kx, %ky) = (%c_0, %c_0) to (%output_dim, %output_dim) {
      %1 = affine.load %arg0[%kx, %ky] : memref<?x?xf32>
      %2 = affine.load %arg1[%kx, %ky] : memref<?x?xf32>
      %3 = arith.mulf %1, %2 : f32
      affine.store %3, %arg2[%kx, %ky] : memref<?x?xf32>
  }
  return
}
// CHECK-LABEL: func @affine_parallel_simple_dynamic_bounds
// CHECK-SAME:  %[[ARG_0:.*]]: memref<?x?xf32>, %[[ARG_1:.*]]: memref<?x?xf32>, %[[ARG_2:.*]]: memref<?x?xf32>
// CHECK:         %[[DIM_INDEX:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[UPPER:.*]] = memref.dim %[[ARG_0]], %[[DIM_INDEX]] : memref<?x?xf32>
// CHECK-NEXT:    %[[LOWER_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[LOWER_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[STEP_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[STEP_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:    scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[LOWER_1]], %[[LOWER_2]]) to (%[[UPPER]], %[[UPPER]]) step (%[[STEP_1]], %[[STEP_2]]) {
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.load
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.load
// CHECK-NEXT:      %[[PRODUCT:.*]] = arith.mulf
// CHECK-NEXT:      store
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////

func.func @affine_parallel_with_reductions(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> (f32, f32) {
  %0:2 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addf", "mulf") -> (f32, f32) {
            %1 = affine.load %arg0[%kx, %ky] : memref<3x3xf32>
            %2 = affine.load %arg1[%kx, %ky] : memref<3x3xf32>
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %1, %2 : f32
            affine.yield %3, %4 : f32, f32
          }
  return %0#0, %0#1 : f32, f32
}
// CHECK-LABEL: func @affine_parallel_with_reductions
// CHECK:         %[[LOWER_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[UPPER_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[LOWER_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[UPPER_2:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[STEP_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[STEP_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[INIT_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[INIT_2:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %[[RES:.*]] = scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[LOWER_1]], %[[LOWER_2]]) to (%[[UPPER_1]], %[[UPPER_2]]) step (%[[STEP_1]], %[[STEP_2]]) init (%[[INIT_1]], %[[INIT_2]]) -> (f32, f32) {
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.load
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.load
// CHECK-NEXT:      %[[PRODUCT:.*]] = arith.mulf
// CHECK-NEXT:      %[[SUM:.*]] = arith.addf
// CHECK-NEXT:      scf.reduce(%[[PRODUCT]]) : f32 {
// CHECK-NEXT:      ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
// CHECK-NEXT:        %[[RES:.*]] = arith.addf
// CHECK-NEXT:        scf.reduce.return %[[RES]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.reduce(%[[SUM]]) : f32 {
// CHECK-NEXT:      ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
// CHECK-NEXT:        %[[RES:.*]] = arith.mulf
// CHECK-NEXT:        scf.reduce.return %[[RES]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////

func.func @affine_parallel_with_reductions_f64(%arg0: memref<3x3xf64>, %arg1: memref<3x3xf64>) -> (f64, f64) {
  %0:2 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addf", "mulf") -> (f64, f64) {
            %1 = affine.load %arg0[%kx, %ky] : memref<3x3xf64>
            %2 = affine.load %arg1[%kx, %ky] : memref<3x3xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %1, %2 : f64
            affine.yield %3, %4 : f64, f64
          }
  return %0#0, %0#1 : f64, f64
}
// CHECK-LABEL: @affine_parallel_with_reductions_f64
// CHECK:  %[[LOWER_1:.*]] = arith.constant 0 : index
// CHECK:  %[[UPPER_1:.*]] = arith.constant 2 : index
// CHECK:  %[[LOWER_2:.*]] = arith.constant 0 : index
// CHECK:  %[[UPPER_2:.*]] = arith.constant 2 : index
// CHECK:  %[[STEP_1:.*]] = arith.constant 1 : index
// CHECK:  %[[STEP_2:.*]] = arith.constant 1 : index
// CHECK:  %[[INIT_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:  %[[INIT_2:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:  %[[RES:.*]] = scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[LOWER_1]], %[[LOWER_2]]) to (%[[UPPER_1]], %[[UPPER_2]]) step (%[[STEP_1]], %[[STEP_2]]) init (%[[INIT_1]], %[[INIT_2]]) -> (f64, f64) {
// CHECK:    %[[VAL_1:.*]] = memref.load
// CHECK:    %[[VAL_2:.*]] = memref.load
// CHECK:    %[[PRODUCT:.*]] = arith.mulf
// CHECK:    %[[SUM:.*]] = arith.addf
// CHECK:    scf.reduce(%[[PRODUCT]]) : f64 {
// CHECK:    ^bb0(%[[LHS:.*]]: f64, %[[RHS:.*]]: f64):
// CHECK:      %[[RES:.*]] = arith.addf
// CHECK:      scf.reduce.return %[[RES]] : f64
// CHECK:    }
// CHECK:    scf.reduce(%[[SUM]]) : f64 {
// CHECK:    ^bb0(%[[LHS:.*]]: f64, %[[RHS:.*]]: f64):
// CHECK:      %[[RES:.*]] = arith.mulf
// CHECK:      scf.reduce.return %[[RES]] : f64
// CHECK:    }
// CHECK:    scf.yield
// CHECK:  }

/////////////////////////////////////////////////////////////////////

func.func @affine_parallel_with_reductions_i64(%arg0: memref<3x3xi64>, %arg1: memref<3x3xi64>) -> (i64, i64) {
  %0:2 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addi", "muli") -> (i64, i64) {
            %1 = affine.load %arg0[%kx, %ky] : memref<3x3xi64>
            %2 = affine.load %arg1[%kx, %ky] : memref<3x3xi64>
            %3 = arith.muli %1, %2 : i64
            %4 = arith.addi %1, %2 : i64
            affine.yield %3, %4 : i64, i64
          }
  return %0#0, %0#1 : i64, i64
}
// CHECK-LABEL: @affine_parallel_with_reductions_i64
// CHECK:  %[[LOWER_1:.*]] = arith.constant 0 : index
// CHECK:  %[[UPPER_1:.*]] = arith.constant 2 : index
// CHECK:  %[[LOWER_2:.*]] = arith.constant 0 : index
// CHECK:  %[[UPPER_2:.*]] = arith.constant 2 : index
// CHECK:  %[[STEP_1:.*]] = arith.constant 1 : index
// CHECK:  %[[STEP_2:.*]] = arith.constant 1 : index
// CHECK:  %[[INIT_1:.*]] = arith.constant 0 : i64
// CHECK:  %[[INIT_2:.*]] = arith.constant 1 : i64
// CHECK:  %[[RES:.*]] = scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[LOWER_1]], %[[LOWER_2]]) to (%[[UPPER_1]], %[[UPPER_2]]) step (%[[STEP_1]], %[[STEP_2]]) init (%[[INIT_1]], %[[INIT_2]]) -> (i64, i64) {
// CHECK:    %[[VAL_1:.*]] = memref.load
// CHECK:    %[[VAL_2:.*]] = memref.load
// CHECK:    %[[PRODUCT:.*]] = arith.muli
// CHECK:    %[[SUM:.*]] = arith.addi
// CHECK:    scf.reduce(%[[PRODUCT]]) : i64 {
// CHECK:    ^bb0(%[[LHS:.*]]: i64, %[[RHS:.*]]: i64):
// CHECK:      %[[RES:.*]] = arith.addi
// CHECK:      scf.reduce.return %[[RES]] : i64
// CHECK:    }
// CHECK:    scf.reduce(%[[SUM]]) : i64 {
// CHECK:    ^bb0(%[[LHS:.*]]: i64, %[[RHS:.*]]: i64):
// CHECK:      %[[RES:.*]] = arith.muli
// CHECK:      scf.reduce.return %[[RES]] : i64
// CHECK:    }
// CHECK:    scf.yield
// CHECK:  }

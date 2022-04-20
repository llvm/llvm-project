// RUN: mlir-opt -allow-unregistered-dialect %s -affine-loop-unroll-jam="unroll-jam-factor=2" | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -affine-loop-unroll-jam="unroll-jam-factor=4" | FileCheck --check-prefix=UJAM-FOUR %s

// CHECK-DAG: [[$MAP_PLUS_1:#map[0-9]+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG: [[$MAP_DIV_OFFSET:#map[0-9]+]] = affine_map<()[s0] -> (((s0 - 1) floordiv 2) * 2 + 1)>
// CHECK-DAG: [[$MAP_SYM_UB:#map[0-9]+]] = affine_map<()[s0, s1] -> (s0, s1, 1024)>

// UJAM-FOUR-DAG: [[$UBMAP:#map[0-9]+]] = affine_map<()[s0] -> (s0 + 8)>
// UJAM-FOUR-DAG: [[$MAP_PLUS_1:#map[0-9]+]] = affine_map<(d0) -> (d0 + 1)>
// UJAM-FOUR-DAG: [[$MAP_PLUS_2:#map[0-9]+]] = affine_map<(d0) -> (d0 + 2)>
// UJAM-FOUR-DAG: [[$MAP_PLUS_3:#map[0-9]+]] = affine_map<(d0) -> (d0 + 3)>

// CHECK-LABEL: func @unroll_jam_imperfect_nest() {
func.func @unroll_jam_imperfect_nest() {
  affine.for %i = 0 to 101 {
    %x = "addi32"(%i, %i) : (index, index) -> i32
    affine.for %j = 0 to 17 {
      %y = "addi32"(%i, %i) : (index, index) -> i32
      %z = "addi32"(%y, %y) : (i32, i32) -> i32
    }
    %w = "foo"(%i, %x) : (index, i32) -> i32
  }
  return
}
// CHECK:      affine.for [[IV0:%arg[0-9]+]] = 0 to 100 step 2 {
// CHECK-NEXT:   [[RES1:%[0-9]+]] = "addi32"([[IV0]], [[IV0]])
// CHECK-NEXT:   [[INC:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   [[RES2:%[0-9]+]] = "addi32"([[INC]], [[INC]])
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 17 {
// CHECK-NEXT:     [[RES3:%[0-9]+]] = "addi32"([[IV0]], [[IV0]])
// CHECK-NEXT:     "addi32"([[RES3]], [[RES3]]) : (i32, i32) -> i32
// CHECK-NEXT:     [[INC1:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:     [[RES4:%[0-9]+]] = "addi32"([[INC1]], [[INC1]])
// CHECK-NEXT:     "addi32"([[RES4]], [[RES4]]) : (i32, i32) -> i32
// CHECK-NEXT:   }
// CHECK:        "foo"([[IV0]], [[RES1]])
// CHECK-NEXT:   affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   "foo"({{.*}}, [[RES2]])
// CHECK:      }
// Cleanup loop (single iteration).
// CHECK:      "addi32"(%c100, %c100)
// CHECK-NEXT: affine.for [[IV0]] = 0 to 17 {
// CHECK-NEXT:   [[RESC:%[0-9]+]] = "addi32"(%c100, %c100)
// CHECK-NEXT:   "addi32"([[RESC]], [[RESC]]) : (i32, i32) -> i32
// CHECK-NEXT: }
// CHECK-NEXT: "foo"(%c100, %{{.*}})
// CHECK-NEXT: return

// CHECK-LABEL: func @loop_nest_unknown_count_1
// CHECK-SAME: [[N:arg[0-9]+]]: index
func.func @loop_nest_unknown_count_1(%N : index) {
  // CHECK-NEXT: affine.for %{{.*}} = 1 to [[$MAP_DIV_OFFSET]]()[%[[N]]] step 2 {
  // CHECK-NEXT:   affine.for %{{.*}} = 1 to 100 {
  // CHECK-NEXT:     "foo"() : () -> i32
  // CHECK-NEXT:     "foo"() : () -> i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // A cleanup loop should be generated here.
  // CHECK-NEXT: affine.for %{{.*}} = [[$MAP_DIV_OFFSET]]()[%[[N]]] to %[[N]] {
  // CHECK-NEXT:   affine.for %{{.*}} = 1 to 100 {
  // CHECK-NEXT:     "foo"() : () -> i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  affine.for %i = 1 to %N {
    affine.for %j = 1 to 100 {
      %x = "foo"() : () -> i32
    }
  }
  return
}

// UJAM-FOUR-LABEL: func @loop_nest_unknown_count_2
// UJAM-FOUR-SAME: %[[N:arg[0-9]+]]: index
func.func @loop_nest_unknown_count_2(%N : index) {
  // UJAM-FOUR-NEXT: affine.for [[IV0:%arg[0-9]+]] = %[[N]] to  [[$UBMAP]]()[%[[N]]] step 4 {
  // UJAM-FOUR-NEXT:   affine.for [[IV1:%arg[0-9]+]] = 1 to 100 {
  // UJAM-FOUR-NEXT:     "foo"([[IV0]])
  // UJAM-FOUR-NEXT:     [[IV_PLUS_1:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
  // UJAM-FOUR-NEXT:     "foo"([[IV_PLUS_1]])
  // UJAM-FOUR-NEXT:     [[IV_PLUS_2:%[0-9]+]] = affine.apply [[$MAP_PLUS_2]]([[IV0]])
  // UJAM-FOUR-NEXT:     "foo"([[IV_PLUS_2]])
  // UJAM-FOUR-NEXT:     [[IV_PLUS_3:%[0-9]+]] = affine.apply [[$MAP_PLUS_3]]([[IV0]])
  // UJAM-FOUR-NEXT:     "foo"([[IV_PLUS_3]])
  // UJAM-FOUR-NEXT:   }
  // UJAM-FOUR-NEXT: }
  // The cleanup loop is a single iteration one and is promoted.
  // UJAM-FOUR-NEXT: [[RES:%[0-9]+]] = affine.apply [[$UBMAP]]()[%[[N]]]
  // UJAM-FOUR-NEXT: affine.for [[IV0]] = 1 to 100 {
  // UJAM-FOUR-NEXT:   "foo"([[RES]])
  // UJAM-FOUR-NEXT: }
  affine.for %i = %N to affine_map<()[s0] -> (s0+9)> ()[%N] {
    affine.for %j = 1 to 100 {
      "foo"(%i) : (index) -> ()
    }
  }
  return
}

// CHECK-LABEL: func @loop_nest_symbolic_and_min_upper_bound
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[N:arg[0-9]+]]: index
// CHECK-SAME: [[K:arg[0-9]+]]: index
func.func @loop_nest_symbolic_and_min_upper_bound(%M : index, %N : index, %K : index) {
  affine.for %i = 0 to min affine_map<()[s0, s1] -> (s0, s1, 1024)>()[%M, %N] {
    affine.for %j = 0 to %K {
      "test.foo"(%i, %j) : (index, index) -> ()
    }
  }
  return
}
// No unroll-and-jam possible here as the lower bound for the cleanup loop won't
// be representable.
// CHECK-NEXT:  affine.for [[IV0:%arg[0-9]+]] = 0 to min #map{{.*}}()[%[[M]], %[[N]]] {
// CHECK-NEXT:    affine.for [[IV1:%arg[0-9]+]] = 0 to %[[K]] {
// CHECK-NEXT:      "test.foo"([[IV0]], [[IV1]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// The inner loop trip count changes each iteration of outer loop.
// Do no unroll-and-jam.
// CHECK-LABEL: func @no_unroll_jam_dependent_ubound
func.func @no_unroll_jam_dependent_ubound(%in0: memref<?xf32, 1>) {
  affine.for %i = 0 to 100 {
    affine.for %k = 0 to affine_map<(d0) -> (d0 + 1)>(%i) {
      %y = "addi32"(%k, %k) : (index, index) -> i32
    }
  }
  return
}
// CHECK:      affine.for [[IV0:%arg[0-9]+]] = 0 to 100 {
// CHECK-NEXT:   affine.for [[IV1:%arg[0-9]+]] = 0 to [[$MAP_PLUS_1]]([[IV0]]) {
// CHECK-NEXT:     "addi32"([[IV1]], [[IV1]])
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: return

// Inner loop with one iter_arg.
// CHECK-LABEL: func @unroll_jam_one_iter_arg
func.func @unroll_jam_one_iter_arg() {
  affine.for %i = 0 to 101 {
    %cst = arith.constant 1 : i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red = affine.for %j = 0 to 17 iter_args(%acc = %cst) -> (i32) {
      %y = "bar"(%i, %j, %acc) : (index, index, i32) -> i32
      affine.yield %y : i32
    }
    %w = "foo"(%i, %x, %red) : (index, i32, i32) -> i32
  }
  return
}
// CHECK:      affine.for [[IV0:%arg[0-9]+]] = 0 to 100 step 2 {
// CHECK-NEXT:   [[CONST1:%[a-zA-Z0-9_]*]] = arith.constant 1 : i32
// CHECK-NEXT:   [[RES1:%[0-9]+]] = "addi32"([[IV0]], [[IV0]])
// CHECK-NEXT:   [[INC:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   [[CONST2:%[a-zA-Z0-9_]*]] = arith.constant 1 : i32
// CHECK-NEXT:   [[RES2:%[0-9]+]] = "addi32"([[INC]], [[INC]])
// CHECK-NEXT:   [[RES3:%[0-9]+]]:2 = affine.for [[IV1:%arg[0-9]+]] = 0 to 17 iter_args([[ACC1:%arg[0-9]+]] = [[CONST1]], [[ACC2:%arg[0-9]+]] = [[CONST2]]) -> (i32, i32) {
// CHECK-NEXT:     [[RES4:%[0-9]+]] = "bar"([[IV0]], [[IV1]], [[ACC1]])
// CHECK-NEXT:     [[INC1:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:     [[RES5:%[0-9]+]] = "bar"([[INC1]], [[IV1]], [[ACC2]])
// CHECK-NEXT:     affine.yield [[RES4]], [[RES5]]
// CHECK-NEXT:   }
// CHECK:        "foo"([[IV0]], [[RES1]], [[RES3]]#0)
// CHECK-NEXT:   affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   "foo"({{.*}}, [[RES2]], [[RES3]]#1)
// CHECK:      }
// Cleanup loop (single iteration).
// CHECK:      constant 1 : i32
// CHECK-NEXT: "addi32"(%c100, %c100)
// CHECK-NEXT: [[RES6:%[0-9]+]] = affine.for
// CHECK-NEXT:   [[RES7:%[0-9]+]] = "bar"(%c100, {{.*}}, {{.*}})
// CHECK-NEXT:   affine.yield [[RES7]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: "foo"(%c100, %{{.*}}, [[RES6]])
// CHECK-NEXT: return

// Inner loop with multiple iter_args.
// CHECK-LABEL: func @unroll_jam_iter_args
func.func @unroll_jam_iter_args() {
  affine.for %i = 0 to 101 {
    %cst = arith.constant 0 : i32
    %cst1 = arith.constant 1 : i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red:2 = affine.for %j = 0 to 17 iter_args(%acc = %cst, %acc1 = %cst1) -> (i32, i32) {
      %y = "bar"(%i, %j, %acc) : (index, index, i32) -> i32
      %z = "bar1"(%i, %j, %acc1) : (index, index, i32) -> i32
      affine.yield %y, %z : i32, i32
    }
    %w = "foo"(%i, %x, %red#0, %red#1) : (index, i32, i32, i32) -> i32
  }
  return
}
// CHECK:      affine.for [[IV0:%arg[0-9]+]] = 0 to 100 step 2 {
// CHECK-NEXT:   [[CONST0:%[a-zA-Z0-9_]*]] = arith.constant 0 : i32
// CHECK-NEXT:   [[CONST1:%[a-zA-Z0-9_]*]] = arith.constant 1 : i32
// CHECK-NEXT:   [[RES1:%[0-9]+]] = "addi32"([[IV0]], [[IV0]])
// CHECK-NEXT:   [[INC:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   [[CONST2:%[a-zA-Z0-9_]*]] = arith.constant 0 : i32
// CHECK-NEXT:   [[CONST3:%[a-zA-Z0-9_]*]] = arith.constant 1 : i32
// CHECK-NEXT:   [[RES2:%[0-9]+]] = "addi32"([[INC]], [[INC]])
// CHECK-NEXT:   [[RES3:%[0-9]+]]:4 = affine.for [[IV1:%arg[0-9]+]] = 0 to 17 iter_args([[ACC0:%arg[0-9]+]] = [[CONST0]], [[ACC1:%arg[0-9]+]] = [[CONST1]],
// CHECK-SAME:   [[ACC2:%arg[0-9]+]] = [[CONST2]], [[ACC3:%arg[0-9]+]] = [[CONST3]]) -> (i32, i32, i32, i32) {
// CHECK-NEXT:     [[RES4:%[0-9]+]] = "bar"([[IV0]], [[IV1]], [[ACC0]])
// CHECK-NEXT:     [[RES5:%[0-9]+]] = "bar1"([[IV0]], [[IV1]], [[ACC1]])
// CHECK-NEXT:     [[INC1:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:     [[RES6:%[0-9]+]] = "bar"([[INC1]], [[IV1]], [[ACC2]])
// CHECK-NEXT:     [[RES7:%[0-9]+]] = "bar1"([[INC1]], [[IV1]], [[ACC3]])
// CHECK-NEXT:     affine.yield [[RES4]], [[RES5]], [[RES6]], [[RES7]]
// CHECK-NEXT:   }
// CHECK:        "foo"([[IV0]], [[RES1]], [[RES3]]#0, [[RES3]]#1)
// CHECK-NEXT:   affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   "foo"({{.*}}, [[RES2]], [[RES3]]#2, [[RES3]]#3)
// CHECK:      }
// Cleanup loop (single iteration).
// CHECK:      constant 0 : i32
// CHECK-NEXT: constant 1 : i32
// CHECK-NEXT: "addi32"(%c100, %c100)
// CHECK-NEXT: [[RES8:%[0-9]+]]:2 = affine.for
// CHECK-NEXT:   [[RES9:%[0-9]+]] = "bar"(%c100, {{.*}}, {{.*}})
// CHECK-NEXT:   [[RES10:%[0-9]+]] = "bar1"(%c100, {{.*}}, {{.*}})
// CHECK-NEXT:   affine.yield [[RES9]], [[RES10]] : i32, i32
// CHECK-NEXT: }
// CHECK-NEXT: "foo"(%c100, %{{.*}}, [[RES8]]#0, [[RES8]]#1)
// CHECK-NEXT: return

// When an iter operand is a function argument, do not replace any use of the
// operand .
// CHECK-LABEL: func @unroll_jam_iter_args_func_arg
// CHECK-SAME:  [[INIT:%arg[0-9]+]]: i32
func.func @unroll_jam_iter_args_func_arg(%in: i32) {
  affine.for %i = 0 to 101 {
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red = affine.for %j = 0 to 17 iter_args(%acc = %in) -> (i32) {
      %y = "bar"(%i, %j, %acc) : (index, index, i32) -> i32
      affine.yield %y : i32
    }
    %w = "foo"(%i, %x, %red) : (index, i32, i32) -> i32
  }
  return
}
// CHECK:      affine.for [[IV0:%arg[0-9]+]] = 0 to 100 step 2 {
// CHECK-NEXT:   [[RES1:%[0-9]+]] = "addi32"([[IV0]], [[IV0]])
// CHECK-NEXT:   [[INC:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   [[RES2:%[0-9]+]] = "addi32"([[INC]], [[INC]])
// CHECK-NEXT:   [[RES3:%[0-9]+]]:2 = affine.for [[IV1:%arg[0-9]+]] = 0 to 17 iter_args([[ACC1:%arg[0-9]+]] = [[INIT]], [[ACC2:%arg[0-9]+]] = [[INIT]]) -> (i32, i32) {
// CHECK-NEXT:     [[RES4:%[0-9]+]] = "bar"([[IV0]], [[IV1]], [[ACC1]])
// CHECK-NEXT:     [[INC1:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:     [[RES5:%[0-9]+]] = "bar"([[INC1]], [[IV1]], [[ACC2]])
// CHECK-NEXT:     affine.yield [[RES4]], [[RES5]]
// CHECK-NEXT:   }
// CHECK:        "foo"([[IV0]], [[RES1]], [[RES3]]#0)
// CHECK-NEXT:   affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   "foo"({{.*}}, [[RES2]], [[RES3]]#1)
// CHECK:      }
// Cleanup loop (single iteration).
// CHECK:      "addi32"(%c100, %c100)
// CHECK-NEXT: [[RES6:%[0-9]+]] = affine.for
// CHECK-NEXT:   [[RES7:%[0-9]+]] = "bar"(%c100, {{.*}}, {{.*}})
// CHECK-NEXT:   affine.yield [[RES7]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: "foo"(%c100, %{{.*}}, [[RES6]])
// CHECK-NEXT: return

// Nested inner loops, each with one iter_arg. The inner most loop uses its
// outer loop's iter_arg as its iter operand.
// CHECK-LABEL: func @unroll_jam_iter_args_nested
func.func @unroll_jam_iter_args_nested() {
  affine.for %i = 0 to 101 {
    %cst = arith.constant 1 : i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red = affine.for %j = 0 to 17 iter_args(%acc = %cst) -> (i32) {
      %red1 = affine.for %k = 0 to 35 iter_args(%acc1 = %acc) -> (i32) {
        %y = "bar"(%i, %j, %k, %acc1) : (index, index, index, i32) -> i32
        affine.yield %y : i32
      }
      affine.yield %red1 : i32
    }
    %w = "foo"(%i, %x, %red) : (index, i32, i32) -> i32
  }
  return
}
// CHECK:      affine.for [[IV0:%arg[0-9]+]] = 0 to 100 step 2 {
// CHECK-NEXT:   [[CONST1:%[a-zA-Z0-9_]*]] = arith.constant 1 : i32
// CHECK-NEXT:   [[RES1:%[0-9]+]] = "addi32"([[IV0]], [[IV0]])
// CHECK-NEXT:   [[INC:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   [[CONST2:%[a-zA-Z0-9_]*]] = arith.constant 1 : i32
// CHECK-NEXT:   [[RES2:%[0-9]+]] = "addi32"([[INC]], [[INC]])
// CHECK-NEXT:   [[RES3:%[0-9]+]]:2 = affine.for [[IV1:%arg[0-9]+]] = 0 to 17 iter_args([[ACC1:%arg[0-9]+]] = [[CONST1]], [[ACC2:%arg[0-9]+]] = [[CONST2]]) -> (i32, i32) {
// CHECK-NEXT:     [[RES4:%[0-9]+]]:2 = affine.for [[IV2:%arg[0-9]+]] = 0 to 35 iter_args([[ACC3:%arg[0-9]+]] = [[ACC1]], [[ACC4:%arg[0-9]+]] = [[ACC2]]) -> (i32, i32) {
// CHECK-NEXT:       [[RES5:%[0-9]+]] = "bar"([[IV0]], [[IV1]], [[IV2]], [[ACC3]])
// CHECK-NEXT:       [[INC1:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:       [[RES6:%[0-9]+]] = "bar"([[INC1]], [[IV1]], [[IV2]], [[ACC4]])
// CHECK-NEXT:       affine.yield [[RES5]], [[RES6]]
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield [[RES4]]#0, [[RES4]]#1
// CHECK-NEXT:   }
// CHECK:        "foo"([[IV0]], [[RES1]], [[RES3]]#0)
// CHECK-NEXT:   affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   "foo"({{.*}}, [[RES2]], [[RES3]]#1)
// CHECK:      }
// Cleanup loop (single iteration).
// CHECK:      constant 1 : i32
// CHECK-NEXT: "addi32"(%c100, %c100)
// CHECK-NEXT: [[RES6:%[0-9]+]] = affine.for
// CHECK-NEXT:   [[RES7:%[0-9]+]] = affine.for
// CHECK-NEXT:     [[RES8:%[0-9]+]] = "bar"(%c100, {{.*}}, {{.*}}, {{.*}})
// CHECK-NEXT:     affine.yield [[RES8]] : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.yield [[RES7]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: "foo"(%c100, %{{.*}}, [[RES6]])
// CHECK-NEXT: return

// Nested inner loops, each with one iter_arg. One loop uses its sibling loop's
// result as its iter operand.
// CHECK-LABEL: func @unroll_jam_iter_args_nested_affine_for_result
func.func @unroll_jam_iter_args_nested_affine_for_result() {
  affine.for %i = 0 to 101 {
    %cst = arith.constant 1 : i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red = affine.for %j = 0 to 17 iter_args(%acc = %cst) -> (i32) {
      %red1 = affine.for %k = 0 to 35 iter_args(%acc1 = %acc) -> (i32) {
        %y = "bar"(%i, %j, %k, %acc1) : (index, index, index, i32) -> i32
        affine.yield %acc : i32
      }
      %red2 = affine.for %l = 0 to 36 iter_args(%acc2 = %red1) -> (i32) {
        %y = "bar"(%i, %j, %l, %acc2) : (index, index, index, i32) -> i32
        affine.yield %y : i32
      }
      affine.yield %red2 : i32
    }
    %w = "foo"(%i, %x, %red) : (index, i32, i32) -> i32
  }
  return
}
// CHECK:      affine.for [[IV0:%arg[0-9]+]] = 0 to 100 step 2 {
// CHECK-NEXT:   [[CONST1:%[a-zA-Z0-9_]*]] = arith.constant 1 : i32
// CHECK-NEXT:   [[RES1:%[0-9]+]] = "addi32"([[IV0]], [[IV0]])
// CHECK-NEXT:   [[INC:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   [[CONST2:%[a-zA-Z0-9_]*]] = arith.constant 1 : i32
// CHECK-NEXT:   [[RES2:%[0-9]+]] = "addi32"([[INC]], [[INC]])
// CHECK-NEXT:   [[RES3:%[0-9]+]]:2 = affine.for [[IV1:%arg[0-9]+]] = 0 to 17 iter_args([[ACC1:%arg[0-9]+]] = [[CONST1]], [[ACC2:%arg[0-9]+]] = [[CONST2]]) -> (i32, i32) {
// CHECK-NEXT:     [[RES4:%[0-9]+]]:2 = affine.for [[IV2:%arg[0-9]+]] = 0 to 35 iter_args([[ACC3:%arg[0-9]+]] = [[ACC1]], [[ACC4:%arg[0-9]+]] = [[ACC2]]) -> (i32, i32) {
// CHECK-NEXT:       [[RES5:%[0-9]+]] = "bar"([[IV0]], [[IV1]], [[IV2]], [[ACC3]])
// CHECK-NEXT:       [[INC1:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:       [[RES6:%[0-9]+]] = "bar"([[INC1]], [[IV1]], [[IV2]], [[ACC4]])
// CHECK-NEXT:       affine.yield [[ACC1]], [[ACC2]]
// CHECK-NEXT:     }
// CHECK-NEXT:     [[RES14:%[0-9]+]]:2 = affine.for [[IV3:%arg[0-9]+]] = 0 to 36 iter_args([[ACC13:%arg[0-9]+]] = [[RES4]]#0, [[ACC14:%arg[0-9]+]] = [[RES4]]#1) -> (i32, i32) {
// CHECK-NEXT:       [[RES15:%[0-9]+]] = "bar"([[IV0]], [[IV1]], [[IV3]], [[ACC13]])
// CHECK-NEXT:       [[INC1:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:       [[RES16:%[0-9]+]] = "bar"([[INC1]], [[IV1]], [[IV3]], [[ACC14]])
// CHECK-NEXT:       affine.yield [[RES15]], [[RES16]]
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield [[RES14]]#0, [[RES14]]#1
// CHECK-NEXT:   }
// CHECK:        "foo"([[IV0]], [[RES1]], [[RES3]]#0)
// CHECK-NEXT:   affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   "foo"({{.*}}, [[RES2]], [[RES3]]#1)
// CHECK:      }
// Cleanup loop (single iteration).
// CHECK:      constant 1 : i32
// CHECK-NEXT: "addi32"(%c100, %c100)
// CHECK-NEXT: [[RES6:%[0-9]+]] = affine.for
// CHECK-NEXT:   [[RES7:%[0-9]+]] = affine.for
// CHECK-NEXT:     [[RES8:%[0-9]+]] = "bar"(%c100, {{.*}}, {{.*}}, {{.*}})
// CHECK-NEXT:     affine.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   [[RES17:%[0-9]+]] = affine.for
// CHECK-NEXT:     [[RES18:%[0-9]+]] = "bar"(%c100, {{.*}}, {{.*}}, {{.*}})
// CHECK-NEXT:     affine.yield [[RES18]] : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.yield [[RES17]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: "foo"(%c100, %{{.*}}, [[RES6]])
// CHECK-NEXT: return

// Nested inner loops, each with one or more iter_args. Yeild the same value
// multiple times.
// CHECK-LABEL: func @unroll_jam_iter_args_nested_yield
func.func @unroll_jam_iter_args_nested_yield() {
  affine.for %i = 0 to 101 {
    %cst = arith.constant 1 : i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red:3 = affine.for %j = 0 to 17 iter_args(%acc = %cst, %acc1 = %cst, %acc2 = %cst) -> (i32, i32, i32) {
      %red1 = affine.for %k = 0 to 35 iter_args(%acc3 = %acc) -> (i32) {
        %y = "bar"(%i, %j, %k, %acc3) : (index, index, index, i32) -> i32
        affine.yield %y : i32
      }
      %red2:2 = affine.for %l = 0 to 36 iter_args(%acc4 = %acc1, %acc5 = %acc2) -> (i32, i32) {
        %y = "bar1"(%i, %j, %l, %acc4, %acc5) : (index, index, index, i32, i32) -> i32
        affine.yield %y, %y : i32, i32
      }
      affine.yield %red1, %red1, %red2#1 : i32, i32, i32
    }
    %w = "foo"(%i, %x, %red#0, %red#2) : (index, i32, i32, i32) -> i32
  }
  return
}
// CHECK:      affine.for [[IV0:%arg[0-9]+]] = 0 to 100 step 2 {
// CHECK-NEXT:   [[CONST1:%[a-zA-Z0-9_]*]] = arith.constant 1 : i32
// CHECK-NEXT:   [[RES1:%[0-9]+]] = "addi32"([[IV0]], [[IV0]])
// CHECK-NEXT:   [[INC:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   [[CONST2:%[a-zA-Z0-9_]*]] = arith.constant 1 : i32
// CHECK-NEXT:   [[RES2:%[0-9]+]] = "addi32"([[INC]], [[INC]])
// CHECK-NEXT:   [[RES3:%[0-9]+]]:6 = affine.for [[IV1:%arg[0-9]+]] = 0 to 17 iter_args([[ACC1:%arg[0-9]+]] = [[CONST1]], [[ACC2:%arg[0-9]+]] = [[CONST1]],
// CHECK-SAME:   [[ACC3:%arg[0-9]+]] = [[CONST1]], [[ACC4:%arg[0-9]+]] = [[CONST2]], [[ACC5:%arg[0-9]+]] = [[CONST2]], [[ACC6:%arg[0-9]+]] = [[CONST2]]) -> (i32, i32, i32, i32, i32, i32) {
// CHECK-NEXT:     [[RES4:%[0-9]+]]:2 = affine.for [[IV2:%arg[0-9]+]] = 0 to 35 iter_args([[ACC7:%arg[0-9]+]] = [[ACC1]], [[ACC8:%arg[0-9]+]] = [[ACC4]]) -> (i32, i32) {
// CHECK-NEXT:       [[RES5:%[0-9]+]] = "bar"([[IV0]], [[IV1]], [[IV2]], [[ACC7]])
// CHECK-NEXT:       [[INC1:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:       [[RES6:%[0-9]+]] = "bar"([[INC1]], [[IV1]], [[IV2]], [[ACC8]])
// CHECK-NEXT:       affine.yield [[RES5]], [[RES6]]
// CHECK-NEXT:     }
// CHECK-NEXT:     [[RES14:%[0-9]+]]:4 = affine.for [[IV3:%arg[0-9]+]] = 0 to 36 iter_args([[ACC13:%arg[0-9]+]] = [[ACC2]], [[ACC14:%arg[0-9]+]] = [[ACC3]],
// CHECK-SAME:     [[ACC15:%arg[0-9]+]] = [[ACC5]], [[ACC16:%arg[0-9]+]] = [[ACC6]]) -> (i32, i32, i32, i32) {
// CHECK-NEXT:       [[RES15:%[0-9]+]] = "bar1"([[IV0]], [[IV1]], [[IV3]], [[ACC13]], [[ACC14]])
// CHECK-NEXT:       [[INC1:%[0-9]+]] = affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:       [[RES16:%[0-9]+]] = "bar1"([[INC1]], [[IV1]], [[IV3]], [[ACC15]], [[ACC16]])
// CHECK-NEXT:       affine.yield [[RES15]], [[RES15]], [[RES16]], [[RES16]]
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield [[RES4]]#0, [[RES4]]#0, [[RES14]]#1, [[RES4]]#1, [[RES4]]#1, [[RES14]]#3
// CHECK-NEXT:   }
// CHECK:        "foo"([[IV0]], [[RES1]], [[RES3]]#0, [[RES3]]#2)
// CHECK-NEXT:   affine.apply [[$MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   "foo"({{.*}}, [[RES2]], [[RES3]]#3, [[RES3]]#5)
// CHECK:      }
// Cleanup loop (single iteration).
// CHECK:      constant 1 : i32
// CHECK-NEXT: "addi32"(%c100, %c100)
// CHECK-NEXT: [[RES6:%[0-9]+]]:3 = affine.for
// CHECK-NEXT:   [[RES7:%[0-9]+]] = affine.for
// CHECK-NEXT:     [[RES8:%[0-9]+]] = "bar"(%c100, {{.*}}, {{.*}}, {{.*}})
// CHECK-NEXT:     affine.yield [[RES8]] : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   [[RES17:%[0-9]+]]:2 = affine.for
// CHECK-NEXT:     [[RES18:%[0-9]+]] = "bar1"(%c100, {{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK-NEXT:     affine.yield [[RES18]], [[RES18]] : i32, i32
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.yield [[RES7]], [[RES7]], [[RES17]]#1 : i32, i32, i32
// CHECK-NEXT: }
// CHECK-NEXT: "foo"(%c100, %{{.*}}, [[RES6]]#0, [[RES6]]#2)
// CHECK-NEXT: return

// CHECK-LABEL: func @unroll_jam_nested_iter_args_mulf
// CHECK-SAME:  [[INIT0:%arg[0-9]+]]: f32, [[INIT1:%arg[0-9]+]]: f32
func.func @unroll_jam_nested_iter_args_mulf(%arg0: memref<21x30xf32, 1>, %init : f32, %init1 : f32) {
  %0 = affine.for %arg3 = 0 to 21 iter_args(%arg4 = %init) -> (f32) {
    %1 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %init1) -> (f32) {
      %3 = affine.load %arg0[%arg3, %arg5] : memref<21x30xf32, 1>
      %4 = arith.addf %arg6, %3 : f32
      affine.yield %4 : f32
    }
    %2 = arith.mulf %arg4, %1 : f32
    affine.yield %2 : f32
  }
  return
}

// CHECK:      %[[CONST0:[a-zA-Z0-9_]*]] = arith.constant 20 : index
// CHECK-NEXT: [[RES:%[0-9]+]]:2 = affine.for %[[IV0:arg[0-9]+]] = 0 to 20 step 2 iter_args([[ACC0:%arg[0-9]+]] = [[INIT0]], [[ACC1:%arg[0-9]+]] = [[INIT0]]) -> (f32, f32) {
// CHECK-NEXT:   [[RES1:%[0-9]+]]:2 = affine.for %[[IV1:arg[0-9]+]] = 0 to 30 iter_args([[ACC2:%arg[0-9]+]] = [[INIT1]], [[ACC3:%arg[0-9]+]] = [[INIT1]]) -> (f32, f32) {
// CHECK-NEXT:     [[LOAD1:%[0-9]+]] = affine.load {{.*}}[%[[IV0]], %[[IV1]]]
// CHECK-NEXT:     [[ADD1:%[0-9]+]] = arith.addf [[ACC2]], [[LOAD1]] : f32
// CHECK-NEXT:     %[[INC1:[0-9]+]] = affine.apply [[$MAP_PLUS_1]](%[[IV0]])
// CHECK-NEXT:     [[LOAD2:%[0-9]+]] = affine.load {{.*}}[%[[INC1]], %[[IV1]]]
// CHECK-NEXT:     [[ADD2:%[0-9]+]] = arith.addf [[ACC3]], [[LOAD2]] : f32
// CHECK-NEXT:     affine.yield [[ADD1]], [[ADD2]]
// CHECK-NEXT:   }
// CHECK-NEXT:   [[MUL1:%[0-9]+]] = arith.mulf [[ACC0]], [[RES1]]#0 : f32
// CHECK-NEXT:   affine.apply
// CHECK-NEXT:   [[MUL2:%[0-9]+]] = arith.mulf [[ACC1]], [[RES1]]#1 : f32
// CHECK-NEXT:   affine.yield [[MUL1]], [[MUL2]]
// CHECK-NEXT: }
// Reduction op.
// CHECK-NEXT: [[MUL3:%[0-9]+]] = arith.mulf [[RES]]#0, [[RES]]#1 : f32
// Cleanup loop (single iteration).
// CHECK-NEXT: [[RES2:%[0-9]+]] = affine.for %[[IV2:arg[0-9]+]] = 0 to 30 iter_args([[ACC4:%arg[0-9]+]] = [[INIT1]]) -> (f32) {
// CHECK-NEXT:   [[LOAD3:%[0-9]+]] = affine.load {{.*}}[%[[CONST0]], %[[IV2]]]
// CHECK-NEXT:   [[ADD3:%[0-9]+]] = arith.addf [[ACC4]], [[LOAD3]] : f32
// CHECK-NEXT:   affine.yield [[ADD3]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: [[MUL4:%[0-9]+]] = arith.mulf [[MUL3]], [[RES2]] : f32
// CHECK-NEXT: return

// CHECK-LABEL: func @unroll_jam_iter_args_addi
// CHECK-SAME:  [[INIT0:%arg[0-9]+]]: i32
func.func @unroll_jam_iter_args_addi(%arg0: memref<21xi32, 1>, %init : i32) {
  %0 = affine.for %arg3 = 0 to 21 iter_args(%arg4 = %init) -> (i32) {
    %1 = affine.load %arg0[%arg3] : memref<21xi32, 1>
    %2 = arith.addi %arg4, %1 : i32
    affine.yield %2 : i32
  }
  return
}

// CHECK:      %[[CONST0:[a-zA-Z0-9_]*]] = arith.constant 20 : index
// CHECK-NEXT: [[RES:%[0-9]+]]:2 = affine.for %[[IV0:arg[0-9]+]] = 0 to 20 step 2 iter_args([[ACC0:%arg[0-9]+]] = [[INIT0]], [[ACC1:%arg[0-9]+]] = [[INIT0]]) -> (i32, i32) {
// CHECK-NEXT:   [[LOAD1:%[0-9]+]] = affine.load {{.*}}[%[[IV0]]]
// CHECK-NEXT:   [[ADD1:%[0-9]+]] = arith.addi [[ACC0]], [[LOAD1]] : i32
// CHECK-NEXT:   %[[INC1:[0-9]+]] = affine.apply [[$MAP_PLUS_1]](%[[IV0]])
// CHECK-NEXT:   [[LOAD2:%[0-9]+]] = affine.load {{.*}}[%[[INC1]]]
// CHECK-NEXT:   [[ADD2:%[0-9]+]] = arith.addi [[ACC1]], [[LOAD2]] : i32
// CHECK-NEXT:   affine.yield [[ADD1]], [[ADD2]]
// CHECK-NEXT: }
// Reduction op.
// CHECK-NEXT: [[ADD3:%[0-9]+]] = arith.addi [[RES]]#0, [[RES]]#1 : i32
// Cleanup loop (single iteration).
// CHECK-NEXT: [[LOAD3:%[0-9]+]] = affine.load {{.*}}[%[[CONST0]]]
// CHECK-NEXT: [[ADD4:%[0-9]+]] = arith.addi [[ADD3]], [[LOAD3]] : i32
// CHECK-NEXT: return

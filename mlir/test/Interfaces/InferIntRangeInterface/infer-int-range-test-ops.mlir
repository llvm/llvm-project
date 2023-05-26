// RUN: mlir-opt -test-int-range-inference %s | FileCheck %s

// CHECK-LABEL: func @constant
// CHECK: %[[cst:.*]] = "test.constant"() <{value = 3 : index}
// CHECK: return %[[cst]]
func.func @constant() -> index {
  %0 = test.with_bounds { umin = 3 : index, umax = 3 : index,
                               smin = 3 : index, smax = 3 : index}
  func.return %0 : index
}

// CHECK-LABEL: func @increment
// CHECK: %[[cst:.*]] = "test.constant"() <{value = 4 : index}
// CHECK: return %[[cst]]
func.func @increment() -> index {
  %0 = test.with_bounds { umin = 3 : index, umax = 3 : index, smin = 0 : index, smax = 0x7fffffffffffffff : index }
  %1 = test.increment %0
  func.return %1 : index
}

// CHECK-LABEL: func @maybe_increment
// CHECK: test.reflect_bounds {smax = 4 : index, smin = 3 : index, umax = 4 : index, umin = 3 : index}
func.func @maybe_increment(%arg0 : i1) -> index {
  %0 = test.with_bounds { umin = 3 : index, umax = 3 : index,
                               smin = 3 : index, smax = 3 : index}
  %1 = scf.if %arg0 -> index {
    scf.yield %0 : index
  } else {
    %2 = test.increment %0
    scf.yield %2 : index
  }
  %3 = test.reflect_bounds %1
  func.return %3 : index
}

// CHECK-LABEL: func @maybe_increment_br
// CHECK: test.reflect_bounds {smax = 4 : index, smin = 3 : index, umax = 4 : index, umin = 3 : index}
func.func @maybe_increment_br(%arg0 : i1) -> index {
  %0 = test.with_bounds { umin = 3 : index, umax = 3 : index,
                               smin = 3 : index, smax = 3 : index}
  cf.cond_br %arg0, ^bb0, ^bb1
^bb0:
    %1 = test.increment %0
    cf.br ^bb2(%1 : index)
^bb1:
    cf.br ^bb2(%0 : index)
^bb2(%2 : index):
  %3 = test.reflect_bounds %2
  func.return %3 : index
}

// CHECK-LABEL: func @for_bounds
// CHECK: test.reflect_bounds {smax = 1 : index, smin = 0 : index, umax = 1 : index, umin = 0 : index}
func.func @for_bounds() -> index {
  %c0 = test.with_bounds { umin = 0 : index, umax = 0 : index,
                                smin = 0 : index, smax = 0 : index}
  %c1 = test.with_bounds { umin = 1 : index, umax = 1 : index,
                                smin = 1 : index, smax = 1 : index}
  %c2 = test.with_bounds { umin = 2 : index, umax = 2 : index,
                                smin = 2 : index, smax = 2 : index}

  %0 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg2 = %c0) -> index {
    scf.yield %arg0 : index
  }
  %1 = test.reflect_bounds %0
  func.return %1 : index
}

// CHECK-LABEL: func @no_analysis_of_loop_variants
// CHECK: test.reflect_bounds {smax = 9223372036854775807 : index, smin = -9223372036854775808 : index, umax = -1 : index, umin = 0 : index}
func.func @no_analysis_of_loop_variants() -> index {
  %c0 = test.with_bounds { umin = 0 : index, umax = 0 : index,
                                smin = 0 : index, smax = 0 : index}
  %c1 = test.with_bounds { umin = 1 : index, umax = 1 : index,
                                smin = 1 : index, smax = 1 : index}
  %c2 = test.with_bounds { umin = 2 : index, umax = 2 : index,
                                smin = 2 : index, smax = 2 : index}

  %0 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg2 = %c0) -> index {
    %1 = test.increment %arg2
    scf.yield %1 : index
  }
  %2 = test.reflect_bounds %0
  func.return %2 : index
}

// CHECK-LABEL: func @region_args
// CHECK: test.reflect_bounds {smax = 4 : index, smin = 3 : index, umax = 4 : index, umin = 3 : index}
func.func @region_args() {
  test.with_bounds_region { umin = 3 : index, umax = 4 : index,
                            smin = 3 : index, smax = 4 : index } %arg0 {
    %0 = test.reflect_bounds %arg0
  }
  func.return
}

// CHECK-LABEL: func @func_args_unbound
// CHECK: test.reflect_bounds {smax = 9223372036854775807 : index, smin = -9223372036854775808 : index, umax = -1 : index, umin = 0 : index}
func.func @func_args_unbound(%arg0 : index) -> index {
  %0 = test.reflect_bounds %arg0
  func.return %0 : index
}

// CHECK-LABEL: func @propagate_across_while_loop_false()
func.func @propagate_across_while_loop_false() -> index {
  // CHECK-DAG: %[[C0:.*]] = "test.constant"() <{value = 0
  // CHECK-DAG: %[[C1:.*]] = "test.constant"() <{value = 1
  %0 = test.with_bounds { umin = 0 : index, umax = 0 : index,
                          smin = 0 : index, smax = 0 : index }
  %1 = scf.while : () -> index {
    %false = arith.constant false
    // CHECK: scf.condition(%{{.*}}) %[[C0]]
    scf.condition(%false) %0 : index
  } do {
  ^bb0(%i1: index):
    scf.yield
  }
  // CHECK: return %[[C1]]
  %2 = test.increment %1
  return %2 : index
}

// CHECK-LABEL: func @propagate_across_while_loop
func.func @propagate_across_while_loop(%arg0 : i1) -> index {
  // CHECK-DAG: %[[C0:.*]] = "test.constant"() <{value = 0
  // CHECK-DAG: %[[C1:.*]] = "test.constant"() <{value = 1
  %0 = test.with_bounds { umin = 0 : index, umax = 0 : index,
                          smin = 0 : index, smax = 0 : index }
  %1 = scf.while : () -> index {
    // CHECK: scf.condition(%{{.*}}) %[[C0]]
    scf.condition(%arg0) %0 : index
  } do {
  ^bb0(%i1: index):
    scf.yield
  }
  // CHECK: return %[[C1]]
  %2 = test.increment %1
  return %2 : index
}

// CHECK-LABEL: func @dont_propagate_across_infinite_loop()
func.func @dont_propagate_across_infinite_loop() -> index {
  // CHECK: %[[C0:.*]] = "test.constant"() <{value = 0
  %0 = test.with_bounds { umin = 0 : index, umax = 0 : index,
                          smin = 0 : index, smax = 0 : index }
  // CHECK: %[[loopRes:.*]] = scf.while
  %1 = scf.while : () -> index {
    %true = arith.constant true
    // CHECK: scf.condition(%{{.*}}) %[[C0]]
    scf.condition(%true) %0 : index
  } do {
  ^bb0(%i1: index):
    scf.yield
  }
  // CHECK: %[[ret:.*]] = test.reflect_bounds %[[loopRes]]
  %2 = test.reflect_bounds %1
  // CHECK: return %[[ret]]
  return %2 : index
}


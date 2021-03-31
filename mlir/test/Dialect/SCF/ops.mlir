// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

func @std_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    scf.for %i1 = %arg0 to %arg1 step %arg2 {
      %min_cmp = cmpi slt, %i0, %i1 : index
      %min = select %min_cmp, %i0, %i1 : index
      %max_cmp = cmpi sge, %i0, %i1 : index
      %max = select %max_cmp, %i0, %i1 : index
      scf.for %i2 = %min to %max step %i1 {
      }
    }
  }
  return
}
// CHECK-LABEL: func @std_for(
//  CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:       %{{.*}} = cmpi slt, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = cmpi sge, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {

func @std_if(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
    %0 = addf %arg1, %arg1 : f32
  }
  return
}
// CHECK-LABEL: func @std_if(
//  CHECK-NEXT:   scf.if %{{.*}} {
//  CHECK-NEXT:     %{{.*}} = addf %{{.*}}, %{{.*}} : f32

func @std_if_else(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
    %0 = addf %arg1, %arg1 : f32
  } else {
    %1 = addf %arg1, %arg1 : f32
  }
  return
}
// CHECK-LABEL: func @std_if_else(
//  CHECK-NEXT:   scf.if %{{.*}} {
//  CHECK-NEXT:     %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     %{{.*}} = addf %{{.*}}, %{{.*}} : f32

func @std_parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                        %arg3 : index, %arg4 : index) {
  %step = constant 1 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step) {
    %min_cmp = cmpi slt, %i0, %i1 : index
    %min = select %min_cmp, %i0, %i1 : index
    %max_cmp = cmpi sge, %i0, %i1 : index
    %max = select %max_cmp, %i0, %i1 : index
    %zero = constant 0.0 : f32
    %int_zero = constant 0 : i32
    %red:2 = scf.parallel (%i2) = (%min) to (%max) step (%i1)
                                      init (%zero, %int_zero) -> (f32, i32) {
      %one = constant 1.0 : f32
      scf.reduce(%one) : f32 {
        ^bb0(%lhs : f32, %rhs: f32):
          %res = addf %lhs, %rhs : f32
          scf.reduce.return %res : f32
      }
      %int_one = constant 1 : i32
      scf.reduce(%int_one) : i32 {
        ^bb0(%lhs : i32, %rhs: i32):
          %res = muli %lhs, %rhs : i32
          scf.reduce.return %res : i32
      }
    }
  }
  return
}
// CHECK-LABEL: func @std_parallel_loop(
//  CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG3:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG4:[A-Za-z0-9]+]]:
//       CHECK:   %[[STEP:.*]] = constant 1 : index
//  CHECK-NEXT:   scf.parallel (%[[I0:.*]], %[[I1:.*]]) = (%[[ARG0]], %[[ARG1]]) to
//       CHECK:   (%[[ARG2]], %[[ARG3]]) step (%[[ARG4]], %[[STEP]]) {
//  CHECK-NEXT:     %[[MIN_CMP:.*]] = cmpi slt, %[[I0]], %[[I1]] : index
//  CHECK-NEXT:     %[[MIN:.*]] = select %[[MIN_CMP]], %[[I0]], %[[I1]] : index
//  CHECK-NEXT:     %[[MAX_CMP:.*]] = cmpi sge, %[[I0]], %[[I1]] : index
//  CHECK-NEXT:     %[[MAX:.*]] = select %[[MAX_CMP]], %[[I0]], %[[I1]] : index
//  CHECK-NEXT:     %[[ZERO:.*]] = constant 0.000000e+00 : f32
//  CHECK-NEXT:     %[[INT_ZERO:.*]] = constant 0 : i32
//  CHECK-NEXT:     scf.parallel (%{{.*}}) = (%[[MIN]]) to (%[[MAX]])
//  CHECK-SAME:          step (%[[I1]])
//  CHECK-SAME:          init (%[[ZERO]], %[[INT_ZERO]]) -> (f32, i32) {
//  CHECK-NEXT:       %[[ONE:.*]] = constant 1.000000e+00 : f32
//  CHECK-NEXT:       scf.reduce(%[[ONE]]) : f32 {
//  CHECK-NEXT:       ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
//  CHECK-NEXT:         %[[RES:.*]] = addf %[[LHS]], %[[RHS]] : f32
//  CHECK-NEXT:         scf.reduce.return %[[RES]] : f32
//  CHECK-NEXT:       }
//  CHECK-NEXT:       %[[INT_ONE:.*]] = constant 1 : i32
//  CHECK-NEXT:       scf.reduce(%[[INT_ONE]]) : i32 {
//  CHECK-NEXT:       ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32):
//  CHECK-NEXT:         %[[RES:.*]] = muli %[[LHS]], %[[RHS]] : i32
//  CHECK-NEXT:         scf.reduce.return %[[RES]] : i32
//  CHECK-NEXT:       }
//  CHECK-NEXT:       scf.yield
//  CHECK-NEXT:     }
//  CHECK-NEXT:     scf.yield

func @parallel_explicit_yield(
    %arg0: index, %arg1: index, %arg2: index) {
  scf.parallel (%i0) = (%arg0) to (%arg1) step (%arg2) {
    scf.yield
  }
  return
}

// CHECK-LABEL: func @parallel_explicit_yield(
//  CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]:
//  CHECK-NEXT: scf.parallel (%{{.*}}) = (%[[ARG0]]) to (%[[ARG1]]) step (%[[ARG2]])
//  CHECK-NEXT: scf.yield
//  CHECK-NEXT: }
//  CHECK-NEXT: return
//  CHECK-NEXT: }

func @std_if_yield(%arg0: i1, %arg1: f32)
{
  %x, %y = scf.if %arg0 -> (f32, f32) {
    %0 = addf %arg1, %arg1 : f32
    %1 = subf %arg1, %arg1 : f32
    scf.yield %0, %1 : f32, f32
  } else {
    %0 = subf %arg1, %arg1 : f32
    %1 = addf %arg1, %arg1 : f32
    scf.yield %0, %1 : f32, f32
  }
  return
}
// CHECK-LABEL: func @std_if_yield(
//  CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
//  CHECK-NEXT: %{{.*}}:2 = scf.if %[[ARG0]] -> (f32, f32) {
//  CHECK-NEXT: %[[T1:.*]] = addf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: %[[T2:.*]] = subf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: scf.yield %[[T1]], %[[T2]] : f32, f32
//  CHECK-NEXT: } else {
//  CHECK-NEXT: %[[T3:.*]] = subf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: %[[T4:.*]] = addf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: scf.yield %[[T3]], %[[T4]] : f32, f32
//  CHECK-NEXT: }

func @std_for_yield(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = constant 0.0 : f32
  %result = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0) -> (f32) {
    %sn = addf %si, %si : f32
    scf.yield %sn : f32
  }
  return
}
// CHECK-LABEL: func @std_for_yield(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]:
// CHECK-NEXT: %[[INIT:.*]] = constant
// CHECK-NEXT: %{{.*}} = scf.for %{{.*}} = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
// CHECK-SAME: iter_args(%[[ITER:.*]] = %[[INIT]]) -> (f32) {
// CHECK-NEXT: %[[NEXT:.*]] = addf %[[ITER]], %[[ITER]] : f32
// CHECK-NEXT: scf.yield %[[NEXT]] : f32
// CHECK-NEXT: }


func @std_for_yield_multi(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = constant 0.0 : f32
  %t0 = constant 1 : i32
  %u0 = constant 1.0 : f32
  %result1:3 = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0, %ti = %t0, %ui = %u0) -> (f32, i32, f32) {
    %sn = addf %si, %si : f32
    %tn = addi %ti, %ti : i32
    %un = subf %ui, %ui : f32
    scf.yield %sn, %tn, %un : f32, i32, f32
  }
  return
}
// CHECK-LABEL: func @std_for_yield_multi(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]:
// CHECK-NEXT: %[[INIT1:.*]] = constant
// CHECK-NEXT: %[[INIT2:.*]] = constant
// CHECK-NEXT: %[[INIT3:.*]] = constant
// CHECK-NEXT: %{{.*}}:3 = scf.for %{{.*}} = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
// CHECK-SAME: iter_args(%[[ITER1:.*]] = %[[INIT1]], %[[ITER2:.*]] = %[[INIT2]], %[[ITER3:.*]] = %[[INIT3]]) -> (f32, i32, f32) {
// CHECK-NEXT: %[[NEXT1:.*]] = addf %[[ITER1]], %[[ITER1]] : f32
// CHECK-NEXT: %[[NEXT2:.*]] = addi %[[ITER2]], %[[ITER2]] : i32
// CHECK-NEXT: %[[NEXT3:.*]] = subf %[[ITER3]], %[[ITER3]] : f32
// CHECK-NEXT: scf.yield %[[NEXT1]], %[[NEXT2]], %[[NEXT3]] : f32, i32, f32


func @conditional_reduce(%buffer: memref<1024xf32>, %lb: index, %ub: index, %step: index) -> (f32) {
  %sum_0 = constant 0.0 : f32
  %c0 = constant 0.0 : f32
  %sum = scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_0) -> (f32) {
	  %t = memref.load %buffer[%iv] : memref<1024xf32>
	  %cond = cmpf ugt, %t, %c0 : f32
	  %sum_next = scf.if %cond -> (f32) {
	    %new_sum = addf %sum_iter, %t : f32
      scf.yield %new_sum : f32
	  } else {
  		scf.yield %sum_iter : f32
	  }
    scf.yield %sum_next : f32
  }
  return %sum : f32
}
// CHECK-LABEL: func @conditional_reduce(
//  CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]
//  CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]
//  CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]
//  CHECK-SAME: %[[ARG3:[A-Za-z0-9]+]]
//  CHECK-NEXT: %[[INIT:.*]] = constant
//  CHECK-NEXT: %[[ZERO:.*]] = constant
//  CHECK-NEXT: %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[ARG1]] to %[[ARG2]] step %[[ARG3]]
//  CHECK-SAME: iter_args(%[[ITER:.*]] = %[[INIT]]) -> (f32) {
//  CHECK-NEXT: %[[T:.*]] = memref.load %[[ARG0]][%[[IV]]]
//  CHECK-NEXT: %[[COND:.*]] = cmpf ugt, %[[T]], %[[ZERO]]
//  CHECK-NEXT: %[[IFRES:.*]] = scf.if %[[COND]] -> (f32) {
//  CHECK-NEXT: %[[THENRES:.*]] = addf %[[ITER]], %[[T]]
//  CHECK-NEXT: scf.yield %[[THENRES]] : f32
//  CHECK-NEXT: } else {
//  CHECK-NEXT: scf.yield %[[ITER]] : f32
//  CHECK-NEXT: }
//  CHECK-NEXT: scf.yield %[[IFRES]] : f32
//  CHECK-NEXT: }
//  CHECK-NEXT: return %[[RESULT]]

// CHECK-LABEL: @while
func @while() {
  %0 = "test.get_some_value"() : () -> i32
  %1 = "test.get_some_value"() : () -> f32

  // CHECK: = scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, f32) -> (i64, f64) {
  %2:2 = scf.while (%arg0 = %0, %arg1 = %1) : (i32, f32) -> (i64, f64) {
    %3:2 = "test.some_operation"(%arg0, %arg1) : (i32, f32) -> (i64, f64)
    %4 = "test.some_condition"(%arg0, %arg1) : (i32, f32) -> i1
    // CHECK: scf.condition(%{{.*}}) %{{.*}}, %{{.*}} : i64, f64
    scf.condition(%4) %3#0, %3#1 : i64, f64
  // CHECK: } do {
  } do {
  // CHECK: ^{{.*}}(%{{.*}}: i64, %{{.*}}: f64):
  ^bb0(%arg2: i64, %arg3: f64):
    %5:2 = "test.some_operation"(%arg2, %arg3): (i64, f64) -> (i32, f32)
    // CHECK: scf.yield %{{.*}}, %{{.*}} : i32, f32
    scf.yield %5#0, %5#1 : i32, f32
  // CHECK: attributes {foo = "bar"}
  } attributes {foo="bar"}
  return
}

// CHECK-LABEL: @infinite_while
func @infinite_while() {
  %true = constant true

  // CHECK: scf.while  : () -> () {
  scf.while : () -> () {
    // CHECK: scf.condition(%{{.*}})
    scf.condition(%true)
  // CHECK: } do {
  } do {
    // CHECK: scf.yield
    scf.yield
  }
  return
}

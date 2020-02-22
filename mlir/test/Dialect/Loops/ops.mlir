// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

func @std_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  loop.for %i0 = %arg0 to %arg1 step %arg2 {
    loop.for %i1 = %arg0 to %arg1 step %arg2 {
      %min_cmp = cmpi "slt", %i0, %i1 : index
      %min = select %min_cmp, %i0, %i1 : index
      %max_cmp = cmpi "sge", %i0, %i1 : index
      %max = select %max_cmp, %i0, %i1 : index
      loop.for %i2 = %min to %max step %i1 {
      }
    }
  }
  return
}
// CHECK-LABEL: func @std_for(
//  CHECK-NEXT:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:       %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {

func @std_if(%arg0: i1, %arg1: f32) {
  loop.if %arg0 {
    %0 = addf %arg1, %arg1 : f32
  }
  return
}
// CHECK-LABEL: func @std_if(
//  CHECK-NEXT:   loop.if %{{.*}} {
//  CHECK-NEXT:     %{{.*}} = addf %{{.*}}, %{{.*}} : f32

func @std_if_else(%arg0: i1, %arg1: f32) {
  loop.if %arg0 {
    %0 = addf %arg1, %arg1 : f32
  } else {
    %1 = addf %arg1, %arg1 : f32
  }
  return
}
// CHECK-LABEL: func @std_if_else(
//  CHECK-NEXT:   loop.if %{{.*}} {
//  CHECK-NEXT:     %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     %{{.*}} = addf %{{.*}}, %{{.*}} : f32

func @std_parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                        %arg3 : index, %arg4 : index) {
  %step = constant 1 : index
  loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step) {
    %min_cmp = cmpi "slt", %i0, %i1 : index
    %min = select %min_cmp, %i0, %i1 : index
    %max_cmp = cmpi "sge", %i0, %i1 : index
    %max = select %max_cmp, %i0, %i1 : index
    %red = loop.parallel (%i2) = (%min) to (%max) step (%i1) {
      %zero = constant 0.0 : f32
      loop.reduce(%zero) {
        ^bb0(%lhs : f32, %rhs: f32):
          %res = addf %lhs, %rhs : f32
          loop.reduce.return %res : f32
      } : f32
    } : f32
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
//  CHECK-NEXT:   loop.parallel (%[[I0:.*]], %[[I1:.*]]) = (%[[ARG0]], %[[ARG1]]) to
//       CHECK:   (%[[ARG2]], %[[ARG3]]) step (%[[ARG4]], %[[STEP]]) {
//  CHECK-NEXT:     %[[MIN_CMP:.*]] = cmpi "slt", %[[I0]], %[[I1]] : index
//  CHECK-NEXT:     %[[MIN:.*]] = select %[[MIN_CMP]], %[[I0]], %[[I1]] : index
//  CHECK-NEXT:     %[[MAX_CMP:.*]] = cmpi "sge", %[[I0]], %[[I1]] : index
//  CHECK-NEXT:     %[[MAX:.*]] = select %[[MAX_CMP]], %[[I0]], %[[I1]] : index
//  CHECK-NEXT:     loop.parallel (%{{.*}}) = (%[[MIN]]) to (%[[MAX]]) step (%[[I1]]) {
//  CHECK-NEXT:       %[[ZERO:.*]] = constant 0.000000e+00 : f32
//  CHECK-NEXT:       loop.reduce(%[[ZERO]]) {
//  CHECK-NEXT:       ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
//  CHECK-NEXT:         %[[RES:.*]] = addf %[[LHS]], %[[RHS]] : f32
//  CHECK-NEXT:         loop.reduce.return %[[RES]] : f32
//  CHECK-NEXT:       } : f32
//  CHECK-NEXT:       loop.yield
//  CHECK-NEXT:     } : f32
//  CHECK-NEXT:     loop.yield

func @parallel_explicit_yield(
    %arg0: index, %arg1: index, %arg2: index) {
  loop.parallel (%i0) = (%arg0) to (%arg1) step (%arg2) {
    loop.yield
  }
  return
}

// CHECK-LABEL: func @parallel_explicit_yield(
//  CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]:
//  CHECK-NEXT: loop.parallel (%{{.*}}) = (%[[ARG0]]) to (%[[ARG1]]) step (%[[ARG2]])
//  CHECK-NEXT: loop.yield
//  CHECK-NEXT: }
//  CHECK-NEXT: return
//  CHECK-NEXT: }

func @std_if_yield(%arg0: i1, %arg1: f32)
{
  %x, %y = loop.if %arg0 -> (f32, f32) {
    %0 = addf %arg1, %arg1 : f32
    %1 = subf %arg1, %arg1 : f32
    loop.yield %0, %1 : f32, f32
  } else {
    %0 = subf %arg1, %arg1 : f32
    %1 = addf %arg1, %arg1 : f32
    loop.yield %0, %1 : f32, f32
  }
  return
}
// CHECK-LABEL: func @std_if_yield(
//  CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
//  CHECK-NEXT: %{{.*}}:2 = loop.if %[[ARG0]] -> (f32, f32) {
//  CHECK-NEXT: %[[T1:.*]] = addf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: %[[T2:.*]] = subf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: loop.yield %[[T1]], %[[T2]] : f32, f32
//  CHECK-NEXT: } else {
//  CHECK-NEXT: %[[T3:.*]] = subf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: %[[T4:.*]] = addf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: loop.yield %[[T3]], %[[T4]] : f32, f32
//  CHECK-NEXT: }

func @std_for_yield(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = constant 0.0 : f32
  %result = loop.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0) -> (f32) {
    %sn = addf %si, %si : f32
    loop.yield %sn : f32
  }
  return
}
// CHECK-LABEL: func @std_for_yield(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]:
// CHECK-NEXT: %[[INIT:.*]] = constant
// CHECK-NEXT: %{{.*}} = loop.for %{{.*}} = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
// CHECK-SAME: iter_args(%[[ITER:.*]] = %[[INIT]]) -> (f32) {
// CHECK-NEXT: %[[NEXT:.*]] = addf %[[ITER]], %[[ITER]] : f32
// CHECK-NEXT: loop.yield %[[NEXT]] : f32
// CHECK-NEXT: }


func @std_for_yield_multi(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = constant 0.0 : f32
  %t0 = constant 1 : i32
  %u0 = constant 1.0 : f32
  %result1:3 = loop.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0, %ti = %t0, %ui = %u0) -> (f32, i32, f32) {
    %sn = addf %si, %si : f32
    %tn = addi %ti, %ti : i32
    %un = subf %ui, %ui : f32
    loop.yield %sn, %tn, %un : f32, i32, f32
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
// CHECK-NEXT: %{{.*}}:3 = loop.for %{{.*}} = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
// CHECK-SAME: iter_args(%[[ITER1:.*]] = %[[INIT1]], %[[ITER2:.*]] = %[[INIT2]], %[[ITER3:.*]] = %[[INIT3]]) -> (f32, i32, f32) {
// CHECK-NEXT: %[[NEXT1:.*]] = addf %[[ITER1]], %[[ITER1]] : f32
// CHECK-NEXT: %[[NEXT2:.*]] = addi %[[ITER2]], %[[ITER2]] : i32
// CHECK-NEXT: %[[NEXT3:.*]] = subf %[[ITER3]], %[[ITER3]] : f32
// CHECK-NEXT: loop.yield %[[NEXT1]], %[[NEXT2]], %[[NEXT3]] : f32, i32, f32


func @conditional_reduce(%buffer: memref<1024xf32>, %lb: index, %ub: index, %step: index) -> (f32) {
  %sum_0 = constant 0.0 : f32
  %c0 = constant 0.0 : f32
  %sum = loop.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_0) -> (f32) {
	  %t = load %buffer[%iv] : memref<1024xf32>
	  %cond = cmpf "ugt", %t, %c0 : f32
	  %sum_next = loop.if %cond -> (f32) {
	    %new_sum = addf %sum_iter, %t : f32
      loop.yield %new_sum : f32
	  } else {
  		loop.yield %sum_iter : f32
	  }
    loop.yield %sum_next : f32
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
//  CHECK-NEXT: %[[RESULT:.*]] = loop.for %[[IV:.*]] = %[[ARG1]] to %[[ARG2]] step %[[ARG3]]
//  CHECK-SAME: iter_args(%[[ITER:.*]] = %[[INIT]]) -> (f32) {
//  CHECK-NEXT: %[[T:.*]] = load %[[ARG0]][%[[IV]]]
//  CHECK-NEXT: %[[COND:.*]] = cmpf "ugt", %[[T]], %[[ZERO]]
//  CHECK-NEXT: %[[IFRES:.*]] = loop.if %[[COND]] -> (f32) {
//  CHECK-NEXT: %[[THENRES:.*]] = addf %[[ITER]], %[[T]]
//  CHECK-NEXT: loop.yield %[[THENRES]] : f32
//  CHECK-NEXT: } else {
//  CHECK-NEXT: loop.yield %[[ITER]] : f32
//  CHECK-NEXT: }
//  CHECK-NEXT: loop.yield %[[IFRES]] : f32
//  CHECK-NEXT: }
//  CHECK-NEXT: return %[[RESULT]]

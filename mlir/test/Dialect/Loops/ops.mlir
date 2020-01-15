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
//  CHECK-NEXT:       "loop.terminator"() : () -> ()
//  CHECK-NEXT:     } : f32
//  CHECK-NEXT:     "loop.terminator"() : () -> ()

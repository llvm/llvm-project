// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-emitc %s | FileCheck %s

func.func @simple_std_for_loop(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = arith.constant 1 : index
  }
  return
}
// CHECK-LABEL: func.func @simple_std_for_loop(
// CHECK-SAME:      %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index) {
// CHECK-NEXT:    emitc.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @simple_std_2_for_loops(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = arith.constant 1 : index
    scf.for %i1 = %arg0 to %arg1 step %arg2 {
      %c1_0 = arith.constant 1 : index
    }
  }
  return
}
// CHECK-LABEL: func.func @simple_std_2_for_loops(
// CHECK-SAME:      %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index) {
// CHECK-NEXT:    emitc.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:      emitc.for %[[VAL_5:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
// CHECK-NEXT:        %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @for_yield(%arg0 : index, %arg1 : index, %arg2 : index) -> (f32, f32) {
  %s0 = arith.constant 0.0 : f32
  %s1 = arith.constant 1.0 : f32
  %result:2 = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0, %sj = %s1) -> (f32, f32) {
    %sn = arith.addf %si, %sj : f32
    scf.yield %sn, %sn : f32, f32
  }
  return %result#0, %result#1 : f32, f32
}
// CHECK-LABEL: func.func @for_yield(
// CHECK-SAME:      %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index) -> (f32, f32) {
// CHECK-NEXT:    %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %[[VAL_5:.*]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:    %[[VAL_6:.*]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:    %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:    memref.store %[[VAL_3]], %[[VAL_5]]{{\[}}%[[VAL_7]]] : memref<1xf32>
// CHECK-NEXT:    memref.store %[[VAL_4]], %[[VAL_6]]{{\[}}%[[VAL_7]]] : memref<1xf32>
// CHECK-NEXT:    emitc.for %[[VAL_8:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
// CHECK-NEXT:      %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_9]]] : memref<1xf32>
// CHECK-NEXT:      %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_9]]] : memref<1xf32>
// CHECK-NEXT:      %[[VAL_12:.*]] = arith.addf %[[VAL_10]], %[[VAL_11]] : f32
// CHECK-NEXT:      %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK-NEXT:      memref.store %[[VAL_12]], %[[VAL_5]]{{\[}}%[[VAL_13]]] : memref<1xf32>
// CHECK-NEXT:      memref.store %[[VAL_12]], %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<1xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[VAL_15:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_14]]] : memref<1xf32>
// CHECK-NEXT:    %[[VAL_16:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_14]]] : memref<1xf32>
// CHECK-NEXT:    return %[[VAL_15]], %[[VAL_16]] : f32, f32
// CHECK-NEXT:  }

func.func @nested_for_yield(%arg0 : index, %arg1 : index, %arg2 : index) -> f32 {
  %s0 = arith.constant 1.0 : f32
  %r = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%iter = %s0) -> (f32) {
    %result = scf.for %i1 = %arg0 to %arg1 step %arg2 iter_args(%si = %iter) -> (f32) {
      %sn = arith.addf %si, %si : f32
      scf.yield %sn : f32
    }
    scf.yield %result : f32
  }
  return %r : f32
}
// CHECK-LABEL: func.func @nested_for_yield(
// CHECK-SAME:      %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index) -> f32 {
// CHECK-NEXT:    %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %[[VAL_4:.*]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:    %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-NEXT:    memref.store %[[VAL_3]], %[[VAL_4]]{{\[}}%[[VAL_5]]] : memref<1xf32>
// CHECK-NEXT:    emitc.for %[[VAL_6:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_7]]] : memref<1xf32>
// CHECK-NEXT:      %[[VAL_9:.*]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:      %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK-NEXT:      memref.store %[[VAL_8]], %[[VAL_9]]{{\[}}%[[VAL_10]]] : memref<1xf32>
// CHECK-NEXT:      emitc.for %[[VAL_11:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
// CHECK-NEXT:        %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_13:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_12]]] : memref<1xf32>
// CHECK-NEXT:        %[[VAL_14:.*]] = arith.addf %[[VAL_13]], %[[VAL_13]] : f32
// CHECK-NEXT:        %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK-NEXT:        memref.store %[[VAL_14]], %[[VAL_9]]{{\[}}%[[VAL_15]]] : memref<1xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_17:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_16]]] : memref<1xf32>
// CHECK-NEXT:      %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK-NEXT:      memref.store %[[VAL_17]], %[[VAL_4]]{{\[}}%[[VAL_18]]] : memref<1xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[VAL_19:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[VAL_20:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_19]]] : memref<1xf32>
// CHECK-NEXT:    return %[[VAL_20]] : f32
// CHECK-NEXT:  }

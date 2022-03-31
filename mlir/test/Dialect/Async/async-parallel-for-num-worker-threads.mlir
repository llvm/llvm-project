// RUN: mlir-opt %s -split-input-file -async-parallel-for=num-workers=-1  \
// RUN: | FileCheck %s --dump-input=always

// CHECK-LABEL: @num_worker_threads(
// CHECK:       %[[MEMREF:.*]]: memref<?xf32>
func @num_worker_threads(%arg0: memref<?xf32>) {

  // CHECK-DAG: %[[scalingCstInit:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-DAG: %[[bracketLowerBound4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[scalingCst4:.*]] = arith.constant 4.000000e+00 : f32
  // CHECK-DAG: %[[bracketLowerBound8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[scalingCst8:.*]] = arith.constant 2.000000e+00 : f32
  // CHECK-DAG: %[[bracketLowerBound16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[scalingCst16:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[bracketLowerBound32:.*]] = arith.constant 32 : index
  // CHECK-DAG: %[[scalingCst32:.*]] = arith.constant 8.000000e-01 : f32
  // CHECK-DAG: %[[bracketLowerBound64:.*]] = arith.constant 64 : index
  // CHECK-DAG: %[[scalingCst64:.*]] = arith.constant 6.000000e-01 : f32
  // CHECK:     %[[workersIndex:.*]] = async.runtime.num_worker_threads : index
  // CHECK:     %[[inBracket4:.*]] = arith.cmpi sgt, %[[workersIndex]], %[[bracketLowerBound4]] : index
  // CHECK:     %[[scalingFactor4:.*]] = arith.select %[[inBracket4]], %[[scalingCst4]], %[[scalingCstInit]] : f32
  // CHECK:     %[[inBracket8:.*]] = arith.cmpi sgt, %[[workersIndex]], %[[bracketLowerBound8]] : index
  // CHECK:     %[[scalingFactor8:.*]] = arith.select %[[inBracket8]], %[[scalingCst8]], %[[scalingFactor4]] : f32
  // CHECK:     %[[inBracket16:.*]] = arith.cmpi sgt, %[[workersIndex]], %[[bracketLowerBound16]] : index
  // CHECK:     %[[scalingFactor16:.*]] = arith.select %[[inBracket16]], %[[scalingCst16]], %[[scalingFactor8]] : f32
  // CHECK:     %[[inBracket32:.*]] = arith.cmpi sgt, %[[workersIndex]], %[[bracketLowerBound32]] : index
  // CHECK:     %[[scalingFactor32:.*]] = arith.select %[[inBracket32]], %[[scalingCst32]], %[[scalingFactor16]] : f32
  // CHECK:     %[[inBracket64:.*]] = arith.cmpi sgt, %[[workersIndex]], %[[bracketLowerBound64]] : index
  // CHECK:     %[[scalingFactor64:.*]] = arith.select %[[inBracket64]], %[[scalingCst64]], %[[scalingFactor32]] : f32
  // CHECK:     %[[workersInt:.*]] = arith.index_cast %[[workersIndex]] : index to i32
  // CHECK:     %[[workersFloat:.*]] = arith.sitofp %[[workersInt]] : i32 to f32
  // CHECK:     %[[scaledFloat:.*]] = arith.mulf %[[scalingFactor64]], %[[workersFloat]] : f32
  // CHECK:     %[[scaledInt:.*]] = arith.fptosi %[[scaledFloat]] : f32 to i32
  // CHECK:     %[[scaledIndex:.*]] = arith.index_cast %[[scaledInt]] : i32 to index

  %lb = arith.constant 0 : index
  %ub = arith.constant 100 : index
  %st = arith.constant 1 : index
  scf.parallel (%i) = (%lb) to (%ub) step (%st) {
    %one = arith.constant 1.0 : f32
    memref.store %one, %arg0[%i] : memref<?xf32>
  }

  return
}

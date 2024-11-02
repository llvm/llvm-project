// RUN: mlir-opt %s -split-input-file -test-vector-linearize | FileCheck %s

// CHECK-LABEL: test_linearize
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>)
//       CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
func.func @test_linearize(%arg0: vector<2x2xf32>) -> vector<2x2xf32> {
//       CHECK: %[[C1:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[C1]] : vector<4xf32> to vector<2x2xf32>

// Arith and math ops are handled in generic way, check some of them
//       CHECK: %{{.*}} =  math.sin %[[ARG]] : vector<4xf32>
  %1 = math.sin %arg0 : vector<2x2xf32>
//       CHECK: %{{.*}} = arith.addf %[[ARG]], %[[C1]] : vector<4xf32>
  %2 = arith.addf %arg0, %0 :  vector<2x2xf32>

//       CHECK: return %[[RES]] : vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

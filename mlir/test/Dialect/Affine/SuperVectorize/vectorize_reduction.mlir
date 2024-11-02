// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=128 test-fastest-varying=0 vectorize-reductions=true" -split-input-file | FileCheck %s

// The inner reduction loop '%j' is vectorized.

func.func @vecdim_reduction(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %red_iter, %ld : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vecdim_reduction
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[red_iter:.*]] = %[[vzero]]) -> (vector<128xf32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[red_iter]], %[[ld]] : vector<128xf32>
// CHECK:           affine.yield %[[add]] : vector<128xf32>
// CHECK:         }
// CHECK:         %[[final_sum:.*]] = vector.reduction <add>, %[[vred:.*]] : vector<128xf32> into f32
// CHECK:         affine.store %[[final_sum]], %{{.*}} : memref<256xf32>
// CHECK:       }

// -----

func.func @vecdim_reduction_minf(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0x7F800000 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %min = arith.minf %red_iter, %ld : f32
     affine.yield %min : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vecdim_reduction_minf
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vmax:.*]] = arith.constant dense<0x7F800000> : vector<128xf32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[red_iter:.*]] = %[[vmax]]) -> (vector<128xf32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[min:.*]] = arith.minf %[[red_iter]], %[[ld]] : vector<128xf32>
// CHECK:           affine.yield %[[min]] : vector<128xf32>
// CHECK:         }
// CHECK:         %[[final_min:.*]] = vector.reduction <minf>, %[[vred:.*]] : vector<128xf32> into f32
// CHECK:         affine.store %[[final_min]], %{{.*}} : memref<256xf32>
// CHECK:       }

// -----

func.func @vecdim_reduction_maxf(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0xFF800000 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %max = arith.maxf %red_iter, %ld : f32
     affine.yield %max : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vecdim_reduction_maxf
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vmin:.*]] = arith.constant dense<0xFF800000> : vector<128xf32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[red_iter:.*]] = %[[vmin]]) -> (vector<128xf32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[max:.*]] = arith.maxf %[[red_iter]], %[[ld]] : vector<128xf32>
// CHECK:           affine.yield %[[max]] : vector<128xf32>
// CHECK:         }
// CHECK:         %[[final_max:.*]] = vector.reduction <maxf>, %[[vred:.*]] : vector<128xf32> into f32
// CHECK:         affine.store %[[final_max]], %{{.*}} : memref<256xf32>
// CHECK:       }

// -----

func.func @vecdim_reduction_minsi(%in: memref<256x512xi32>, %out: memref<256xi32>) {
 %cst = arith.constant 2147483647 : i32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (i32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xi32>
     %min = arith.minsi %red_iter, %ld : i32
     affine.yield %min : i32
   }
   affine.store %final_red, %out[%i] : memref<256xi32>
 }
 return
}

// CHECK-LABEL: @vecdim_reduction_minsi
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vmax:.*]] = arith.constant dense<2147483647> : vector<128xi32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[red_iter:.*]] = %[[vmax]]) -> (vector<128xi32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xi32>, vector<128xi32>
// CHECK:           %[[min:.*]] = arith.minsi %[[red_iter]], %[[ld]] : vector<128xi32>
// CHECK:           affine.yield %[[min]] : vector<128xi32>
// CHECK:         }
// CHECK:         %[[final_min:.*]] = vector.reduction <minsi>, %[[vred:.*]] : vector<128xi32> into i32
// CHECK:         affine.store %[[final_min]], %{{.*}} : memref<256xi32>
// CHECK:       }

// -----

func.func @vecdim_reduction_maxsi(%in: memref<256x512xi32>, %out: memref<256xi32>) {
 %cst = arith.constant -2147483648 : i32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (i32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xi32>
     %max = arith.maxsi %red_iter, %ld : i32
     affine.yield %max : i32
   }
   affine.store %final_red, %out[%i] : memref<256xi32>
 }
 return
}

// CHECK-LABEL: @vecdim_reduction_maxsi
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vmin:.*]] = arith.constant dense<-2147483648> : vector<128xi32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[red_iter:.*]] = %[[vmin]]) -> (vector<128xi32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xi32>, vector<128xi32>
// CHECK:           %[[max:.*]] = arith.maxsi %[[red_iter]], %[[ld]] : vector<128xi32>
// CHECK:           affine.yield %[[max]] : vector<128xi32>
// CHECK:         }
// CHECK:         %[[final_max:.*]] = vector.reduction <maxsi>, %[[vred:.*]] : vector<128xi32> into i32
// CHECK:         affine.store %[[final_max]], %{{.*}} : memref<256xi32>
// CHECK:       }

// -----

func.func @vecdim_reduction_minui(%in: memref<256x512xi32>, %out: memref<256xi32>) {
 %cst = arith.constant -1 : i32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (i32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xi32>
     %min = arith.minui %red_iter, %ld : i32
     affine.yield %min : i32
   }
   affine.store %final_red, %out[%i] : memref<256xi32>
 }
 return
}

// CHECK-LABEL: @vecdim_reduction_minui
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vmax:.*]] = arith.constant dense<-1> : vector<128xi32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[red_iter:.*]] = %[[vmax]]) -> (vector<128xi32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xi32>, vector<128xi32>
// CHECK:           %[[min:.*]] = arith.minui %[[red_iter]], %[[ld]] : vector<128xi32>
// CHECK:           affine.yield %[[min]] : vector<128xi32>
// CHECK:         }
// CHECK:         %[[final_min:.*]] = vector.reduction <minui>, %[[vred:.*]] : vector<128xi32> into i32
// CHECK:         affine.store %[[final_min]], %{{.*}} : memref<256xi32>
// CHECK:       }

// -----

func.func @vecdim_reduction_maxui(%in: memref<256x512xi32>, %out: memref<256xi32>) {
 %cst = arith.constant 0 : i32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (i32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xi32>
     %max = arith.maxui %red_iter, %ld : i32
     affine.yield %max : i32
   }
   affine.store %final_red, %out[%i] : memref<256xi32>
 }
 return
}

// CHECK-LABEL: @vecdim_reduction_maxui
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vmin:.*]] = arith.constant dense<0> : vector<128xi32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[red_iter:.*]] = %[[vmin]]) -> (vector<128xi32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xi32>, vector<128xi32>
// CHECK:           %[[max:.*]] = arith.maxui %[[red_iter]], %[[ld]] : vector<128xi32>
// CHECK:           affine.yield %[[max]] : vector<128xi32>
// CHECK:         }
// CHECK:         %[[final_max:.*]] = vector.reduction <maxui>, %[[vred:.*]] : vector<128xi32> into i32
// CHECK:         affine.store %[[final_max]], %{{.*}} : memref<256xi32>
// CHECK:       }

// -----

// The inner reduction loop '%j' is vectorized. (The order of addf's operands is
// different than in the previous test case).

func.func @vecdim_reduction_comm(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %ld, %red_iter : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vecdim_reduction_comm
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[red_iter:.*]] = %[[vzero]]) -> (vector<128xf32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[ld]], %[[red_iter]] : vector<128xf32>
// CHECK:           affine.yield %[[add]] : vector<128xf32>
// CHECK:         }
// CHECK:         %[[final_sum:.*]] = vector.reduction <add>, %[[vred:.*]] : vector<128xf32> into f32
// CHECK:         affine.store %[[final_sum]], %{{.*}} : memref<256xf32>
// CHECK:       }

// -----

// The inner reduction loop '%j' is vectorized. Transforming the input before
// performing the accumulation doesn't cause any problem.

func.func @vecdim_reduction_expsin(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %sin = math.sin %ld : f32
     %exp = math.exp %sin : f32
     %add = arith.addf %red_iter, %exp : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vecdim_reduction_expsin
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[red_iter:.*]] = %[[vzero]]) -> (vector<128xf32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[sin:.*]] = math.sin %[[ld]]
// CHECK:           %[[exp:.*]] = math.exp %[[sin]]
// CHECK:           %[[add:.*]] = arith.addf %[[red_iter]], %[[exp]] : vector<128xf32>
// CHECK:           affine.yield %[[add]] : vector<128xf32>
// CHECK:         }
// CHECK:         %[[final_sum:.*]] = vector.reduction <add>, %[[vred:.*]] : vector<128xf32> into f32
// CHECK:         affine.store %[[final_sum]], %{{.*}} : memref<256xf32>
// CHECK:       }

// -----

// Two reductions at the same time. The inner reduction loop '%j' is vectorized.

func.func @two_vecdim_reductions(%in: memref<256x512xf32>, %out_sum: memref<256xf32>, %out_prod: memref<256xf32>) {
 %cst = arith.constant 1.000000e+00 : f32
 affine.for %i = 0 to 256 {
   // Note that we pass the same constant '1.0' as initial values for both
   // reductions.
   %sum, %prod = affine.for %j = 0 to 512 iter_args(%part_sum = %cst, %part_prod = %cst) -> (f32, f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %part_sum, %ld : f32
     %mul = arith.mulf %part_prod, %ld : f32
     affine.yield %add, %mul : f32, f32
   }
   affine.store %sum, %out_sum[%i] : memref<256xf32>
   affine.store %prod, %out_prod[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @two_vecdim_reductions
// CHECK:       %[[cst:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[vone:.*]] = arith.constant dense<1.000000e+00> : vector<128xf32>
// CHECK:         %[[vred:.*]]:2 = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[part_sum:.*]] = %[[vzero]], %[[part_prod:.*]] = %[[vone]]) -> (vector<128xf32>, vector<128xf32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[part_sum]], %[[ld]] : vector<128xf32>
// CHECK:           %[[mul:.*]] = arith.mulf %[[part_prod]], %[[ld]] : vector<128xf32>
// CHECK:           affine.yield %[[add]], %[[mul]] : vector<128xf32>, vector<128xf32>
// CHECK:         }
// CHECK:         %[[nonfinal_sum:.*]] = vector.reduction <add>, %[[vred:.*]]#0 : vector<128xf32> into f32
// Note that to compute the final sum we need to add the original initial value
// (%cst) since it is not zero.
// CHECK:         %[[final_sum:.*]] = arith.addf %[[nonfinal_sum]], %[[cst]] : f32
// For the final product we don't need to do this additional step because the
// initial value equals to 1 (the neutral element for multiplication).
// CHECK:         %[[final_prod:.*]] = vector.reduction <mul>, %[[vred:.*]]#1 : vector<128xf32> into f32
// CHECK:         affine.store %[[final_sum]], %{{.*}} : memref<256xf32>
// CHECK:         affine.store %[[final_prod]], %{{.*}} : memref<256xf32>
// CHECK:       }

// -----

// The integer case.

func.func @two_vecdim_reductions_int(%in: memref<256x512xi64>, %out_sum: memref<256xi64>, %out_prod: memref<256xi64>) {
 %cst0 = arith.constant 0 : i64
 %cst1 = arith.constant 1 : i64
 affine.for %i = 0 to 256 {
   %sum, %prod = affine.for %j = 0 to 512 iter_args(%part_sum = %cst0, %part_prod = %cst1) -> (i64, i64) {
     %ld = affine.load %in[%i, %j] : memref<256x512xi64>
     %add = arith.addi %part_sum, %ld : i64
     %mul = arith.muli %part_prod, %ld : i64
     affine.yield %add, %mul : i64, i64
   }
   affine.store %sum, %out_sum[%i] : memref<256xi64>
   affine.store %prod, %out_prod[%i] : memref<256xi64>
 }
 return
}

// CHECK-LABEL: @two_vecdim_reductions
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vzero:.*]] = arith.constant dense<0> : vector<128xi64>
// CHECK:         %[[vone:.*]] = arith.constant dense<1> : vector<128xi64>
// CHECK:         %[[vred:.*]]:2 = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[part_sum:.*]] = %[[vzero]], %[[part_prod:.*]] = %[[vone]]) -> (vector<128xi64>, vector<128xi64>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xi64>, vector<128xi64>
// CHECK:           %[[add:.*]] = arith.addi %[[part_sum]], %[[ld]] : vector<128xi64>
// CHECK:           %[[mul:.*]] = arith.muli %[[part_prod]], %[[ld]] : vector<128xi64>
// CHECK:           affine.yield %[[add]], %[[mul]] : vector<128xi64>, vector<128xi64>
// CHECK:         }
// CHECK:         %[[final_sum:.*]] = vector.reduction <add>, %[[vred:.*]]#0 : vector<128xi64> into i64
// CHECK:         %[[final_prod:.*]] = vector.reduction <mul>, %[[vred:.*]]#1 : vector<128xi64> into i64
// CHECK:         affine.store %[[final_sum]], %{{.*}} : memref<256xi64>
// CHECK:         affine.store %[[final_prod]], %{{.*}} : memref<256xi64>
// CHECK:       }

// -----

// The outer reduction loop '%j' is vectorized.

func.func @vecdim_reduction_nested(%in: memref<256x512xf32>, %out: memref<1xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 %outer_red = affine.for %j = 0 to 512 iter_args(%outer_iter = %cst) -> (f32) {
   %inner_red = affine.for %i = 0 to 256 iter_args(%inner_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %inner_iter, %ld : f32
     affine.yield %add : f32
   }
   %outer_add = arith.addf %outer_iter, %inner_red : f32
   affine.yield %outer_add : f32
 }
 affine.store %outer_red, %out[0] : memref<1xf32>
 return
}

// CHECK-LABEL: @vecdim_reduction_nested
// CHECK:       %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:       %[[outer_red:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[outer_iter:.*]] = %[[vzero]]) -> (vector<128xf32>) {
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[inner_red:.*]] = affine.for %{{.*}} = 0 to 256 iter_args(%[[inner_iter:.*]] = %[[vzero]]) -> (vector<128xf32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[inner_iter]], %[[ld]] : vector<128xf32>
// CHECK:           affine.yield %[[add]] : vector<128xf32>
// CHECK:         }
// CHECK:         %[[outer_add:.*]] = arith.addf %[[outer_iter]], %[[inner_red]] : vector<128xf32>
// CHECK:         affine.yield %[[outer_add]] : vector<128xf32>
// CHECK:       }
// CHECK:       %[[final_sum:.*]] = vector.reduction <add>, %[[outer_red:.*]] : vector<128xf32> into f32
// CHECK:       affine.store %[[final_sum]], %{{.*}} : memref<1xf32>

// -----

// The inner reduction loop '%j' computes partial sums as a side effect and
// is not vectorized.

func.func @vecdim_partial_sums_1_rejected(%in: memref<256x512xf32>, %out_sum: memref<256xf32>, %out_prod: memref<256xf32>, %out_partsum: memref<256x512xf32>) {
 %cst = arith.constant 1.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %sum, %prod = affine.for %j = 0 to 512 iter_args(%part_sum = %cst, %part_prod = %cst) -> (f32, f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %part_sum, %ld : f32
     %mul = arith.mulf %part_prod, %ld : f32
     affine.store %add, %out_partsum[%i, %j] : memref<256x512xf32>
     affine.yield %add, %mul : f32, f32
   }
   affine.store %sum, %out_sum[%i] : memref<256xf32>
   affine.store %prod, %out_prod[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vecdim_partial_sums_1_rejected
// CHECK-NOT:   vector

// -----

// The inner reduction loop '%j' computes partial sums as a side effect and
// is not vectorized.

func.func @vecdim_partial_sums_2_rejected(%in: memref<256x512xf32>, %out_sum: memref<256xf32>, %out_prod: memref<256xf32>, %out_partsum: memref<256x512xf32>) {
 %cst = arith.constant 1.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %sum, %prod = affine.for %j = 0 to 512 iter_args(%part_sum = %cst, %part_prod = %cst) -> (f32, f32) {
     affine.store %part_sum, %out_partsum[%i, %j] : memref<256x512xf32>
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %part_sum, %ld : f32
     %mul = arith.mulf %part_prod, %ld : f32
     affine.yield %add, %mul : f32, f32
   }
   affine.store %sum, %out_sum[%i] : memref<256xf32>
   affine.store %prod, %out_prod[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vecdim_partial_sums_2_rejected
// CHECK-NOT:   vector

// -----

// The inner reduction loop '%j' performs an unknown reduction operation and is
// not vectorized.

func.func @vecdim_unknown_reduction_rejected(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 1.000000e+00 : f32
 %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
   %add = arith.addf %red_iter, %red_iter : f32
   affine.yield %add : f32
 }
 affine.store %final_red, %out[0] : memref<256xf32>
 return
}

// CHECK-LABEL: @vecdim_unknown_reduction_rejected
// CHECK-NOT:   vector

// -----

// The inner reduction loop '%j' doesn't perform any operation which is not
// recognized as a standard reduction.

func.func @vecdim_none_reduction_rejected(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 1.000000e+00 : f32
 %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
   affine.yield %red_iter : f32
 }
 affine.store %final_red, %out[0] : memref<256xf32>
 return
}

// CHECK-LABEL: @vecdim_none_reduction_rejected
// CHECK-NOT:   vector

// -----

// The number of iterations is not divisable by the vector size, so a mask has
// to be applied to the last update of the accumulator.

func.func @vecdim_reduction_masked(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 500 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %red_iter, %ld : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK:       #[[$map0:.*]] = affine_map<([[d0:.*]]) -> (-[[d0]] + 500)>
// CHECK-LABEL: @vecdim_reduction_masked
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[vred:.*]] = affine.for %[[iv:.*]] = 0 to 500 step 128 iter_args(%[[red_iter:.*]] = %[[vzero]]) -> (vector<128xf32>) {
// CHECK:           %[[elems_left:.*]] = affine.apply #[[$map0]](%[[iv]])
// CHECK:           %[[mask:.*]] = vector.create_mask %[[elems_left]] : vector<128xi1>
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[select:.*]] = arith.select %[[mask]], %[[ld]], %[[vzero]] : vector<128xi1>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[red_iter]], %[[select]] : vector<128xf32>
// CHECK:           affine.yield %[[add]] : vector<128xf32>
// CHECK:         }
// CHECK:         %[[final_sum:.*]] = vector.reduction <add>, %[[vred:.*]] : vector<128xf32> into f32
// CHECK:         affine.store %[[final_sum]], %{{.*}} : memref<256xf32>
// CHECK:       }

// -----

// The number of iteration is not known, so a mask has to be applied.

func.func @vecdim_reduction_masked_unknown_ub(%in: memref<256x512xf32>, %out: memref<256xf32>, %bnd: index) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to %bnd iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %red_iter, %ld : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK:       #[[$map1:.*]] = affine_map<([[d0:.*]]){{\[}}[[s0:.*]]{{\]}} -> (-[[d0]] + [[s0]])>
// CHECK-LABEL: @vecdim_reduction_masked_unknown_ub
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[vred:.*]] = affine.for %[[iv:.*]] = 0 to %[[bnd:.*]] step 128 iter_args(%[[red_iter:.*]] = %[[vzero]]) -> (vector<128xf32>) {
// CHECK:           %[[elems_left:.*]] = affine.apply #[[$map1]](%[[iv]])[%[[bnd]]]
// CHECK:           %[[mask:.*]] = vector.create_mask %[[elems_left]] : vector<128xi1>
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[select:.*]] = arith.select %[[mask]], %[[ld]], %[[vzero]] : vector<128xi1>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[red_iter]], %[[select]] : vector<128xf32>
// CHECK:           affine.yield %[[add]] : vector<128xf32>
// CHECK:         }
// CHECK:         %[[final_sum:.*]] = vector.reduction <add>, %[[vred:.*]] : vector<128xf32> into f32
// CHECK:         affine.store %[[final_sum]], %{{.*}} : memref<256xf32>
// CHECK:       }

// -----

// The lower bound is nonzero, but the number of iterations is divisible by the
// vector size, so masking is not needed.

func.func @vecdim_reduction_nonzero_lb(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 127 to 511 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %red_iter, %ld : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vecdim_reduction_nonzero_lb
// CHECK:         %{{.*}} = affine.for %{{.*}} = 127 to 511 step 128 iter_args({{.*}}) -> (vector<128xf32>) {
// CHECK-NOT:     vector.create_mask

// -----

// The lower bound is unknown, so we need to create a mask.

func.func @vecdim_reduction_masked_unknown_lb(%in: memref<256x512xf32>, %out: memref<256xf32>, %lb: index) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = %lb to 512 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %red_iter, %ld : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK:       #[[$map2:.*]] = affine_map<([[d0:.*]]) -> (-[[d0]] + 512)>
// CHECK-LABEL: @vecdim_reduction_masked_unknown_lb
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %{{.*}} = affine.for %[[iv:.*]] = %[[lb:.*]] to 512 step 128 iter_args(%[[red_iter:.*]] = {{.*}}) -> (vector<128xf32>) {
// CHECK:           %[[elems_left:.*]] = affine.apply #[[$map2]](%[[iv]])
// CHECK:           %[[mask:.*]] = vector.create_mask %[[elems_left]] : vector<128xi1>
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[select:.*]] = arith.select %[[mask]], %[[ld]], %[[vzero]] : vector<128xi1>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[red_iter]], %[[select]] : vector<128xf32>
// CHECK:           affine.yield %[[add]] : vector<128xf32>

// -----

// The upper bound is a minimum expression.

func.func @vecdim_reduction_complex_ub(%in: memref<256x512xf32>, %out: memref<256xf32>, %M: index, %N: index) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to min affine_map<(d0, d1) -> (d0, d1*2)>(%M, %N) iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %red_iter, %ld : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK:       #[[$map3:.*]] = affine_map<([[d0:.*]], [[d1:.*]]) -> ([[d0]], [[d1]] * 2)>
// CHECK:       #[[$map3_sub:.*]] = affine_map<([[d0:.*]], [[d1:.*]]) -> ([[d0]] - [[d1]])>
// CHECK-LABEL: @vecdim_reduction_complex_ub
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %{{.*}} = affine.for %[[iv:.*]] = 0 to min #[[$map3]](%[[M:.*]], %[[N:.*]]) step 128 iter_args(%[[red_iter:.*]] = {{.*}}) -> (vector<128xf32>) {
// CHECK:           %[[ub:.*]] = affine.min #[[$map3]](%[[M]], %[[N]])
// CHECK:           %[[elems_left:.*]] = affine.apply #[[$map3_sub]](%[[ub]], %[[iv]])
// CHECK:           %[[mask:.*]] = vector.create_mask %[[elems_left]] : vector<128xi1>
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[select:.*]] = arith.select %[[mask]], %[[ld]], %[[vzero]] : vector<128xi1>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[red_iter]], %[[select]] : vector<128xf32>
// CHECK:           affine.yield %[[add]] : vector<128xf32>

// -----

// The same mask is applied to both reductions.

func.func @vecdim_two_reductions_masked(%in: memref<256x512xf32>, %out: memref<512xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_sum, %final_expsum = affine.for %j = 0 to 500 iter_args(%sum_iter = %cst, %expsum_iter = %cst) -> (f32, f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %exp = math.exp %ld : f32
     %add = arith.addf %sum_iter, %ld : f32
     %eadd = arith.addf %expsum_iter, %exp : f32
     affine.yield %add, %eadd : f32, f32
   }
   affine.store %final_sum, %out[2*%i] : memref<512xf32>
   affine.store %final_expsum, %out[2*%i + 1] : memref<512xf32>
 }
 return
}

// CHECK:       #[[$map4:.*]] = affine_map<([[d0:.*]]) -> (-[[d0]] + 500)>
// CHECK-LABEL: @vecdim_two_reductions_masked
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vzero0:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[vzero1:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %{{.*}} = affine.for %[[iv:.*]] = 0 to 500 step 128 iter_args(%[[sum_iter:.*]] = {{.*}}, %[[esum_iter:.*]] = {{.*}}) -> (vector<128xf32>, vector<128xf32>) {
// CHECK:           %[[elems_left:.*]] = affine.apply #[[$map4]](%[[iv]])
// CHECK:           %[[mask:.*]] = vector.create_mask %[[elems_left]] : vector<128xi1>
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[exp:.*]] = math.exp %[[ld]] : vector<128xf32>
// CHECK:           %[[select0:.*]] = arith.select %[[mask]], %[[ld]], %[[vzero0]] : vector<128xi1>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[sum_iter]], %[[select0]] : vector<128xf32>
// CHECK:           %[[select1:.*]] = arith.select %[[mask]], %[[exp]], %[[vzero1]] : vector<128xi1>, vector<128xf32>
// CHECK:           %[[eadd:.*]] = arith.addf %[[esum_iter]], %[[select1]] : vector<128xf32>
// CHECK:           affine.yield %[[add]], %[[eadd]] : vector<128xf32>
// CHECK:         }

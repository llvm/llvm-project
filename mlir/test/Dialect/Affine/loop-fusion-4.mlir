// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{mode=producer}))' -split-input-file | FileCheck %s --check-prefix=PRODUCER-CONSUMER
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{mode=producer fusion-maximal}))' -split-input-file | FileCheck %s --check-prefix=PRODUCER-CONSUMER-MAXIMAL
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{fusion-maximal mode=sibling}))' -split-input-file | FileCheck %s --check-prefix=SIBLING-MAXIMAL
// All fusion: producer-consumer and sibling.
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion))' -split-input-file | FileCheck %s --check-prefix=ALL
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(spirv.func(affine-loop-fusion{mode=producer}))' -split-input-file | FileCheck %s --check-prefix=SPIRV

// Part I of fusion tests in  mlir/test/Transforms/loop-fusion.mlir.
// Part II of fusion tests in mlir/test/Transforms/loop-fusion-2.mlir
// Part III of fusion tests in mlir/test/Transforms/loop-fusion-3.mlir

// Expects fusion of producer into consumer at depth 4 and subsequent removal of
// source loop.
// PRODUCER-CONSUMER-LABEL: func @unflatten4d
func.func @unflatten4d(%arg1: memref<7x8x9x10xf32>) {
  %m = memref.alloc() : memref<5040xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 7 {
    affine.for %i1 = 0 to 8 {
      affine.for %i2 = 0 to 9 {
        affine.for %i3 = 0 to 10 {
          affine.store %cf7, %m[720 * %i0 + 90 * %i1 + 10 * %i2 + %i3] : memref<5040xf32>
        }
      }
    }
  }
  affine.for %i0 = 0 to 7 {
    affine.for %i1 = 0 to 8 {
      affine.for %i2 = 0 to 9 {
        affine.for %i3 = 0 to 10 {
          %v0 = affine.load %m[720 * %i0 + 90 * %i1 + 10 * %i2 + %i3] : memref<5040xf32>
          affine.store %v0, %arg1[%i0, %i1, %i2, %i3] : memref<7x8x9x10xf32>
        }
      }
    }
  }
  return
}

// PRODUCER-CONSUMER:        affine.for
// PRODUCER-CONSUMER-NEXT:     affine.for
// PRODUCER-CONSUMER-NEXT:       affine.for
// PRODUCER-CONSUMER-NEXT:         affine.for
// PRODUCER-CONSUMER-NOT:    affine.for
// PRODUCER-CONSUMER: return

// -----

// Expects fusion of producer into consumer at depth 2 and subsequent removal of
// source loop.
// PRODUCER-CONSUMER-LABEL: func @unflatten2d_with_transpose
func.func @unflatten2d_with_transpose(%arg1: memref<8x7xf32>) {
  %m = memref.alloc() : memref<56xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 7 {
    affine.for %i1 = 0 to 8 {
      affine.store %cf7, %m[8 * %i0 + %i1] : memref<56xf32>
    }
  }
  affine.for %i0 = 0 to 8 {
    affine.for %i1 = 0 to 7 {
      %v0 = affine.load %m[%i0 + 8 * %i1] : memref<56xf32>
      affine.store %v0, %arg1[%i0, %i1] : memref<8x7xf32>
    }
  }
  return
}

// PRODUCER-CONSUMER:        affine.for
// PRODUCER-CONSUMER-NEXT:     affine.for
// PRODUCER-CONSUMER-NOT:    affine.for
// PRODUCER-CONSUMER: return

// -----

// Expects fusion of producer into consumer at depth 1 and source loop to not
// be removed due to difference in loop steps.
// PRODUCER-CONSUMER-LABEL: func @check_src_dst_step
func.func @check_src_dst_step(%m : memref<100xf32>,
                         %src: memref<100xf32>,
                         %out: memref<100xf32>) {
  affine.for %i0 = 0 to 100 {
    %r1 = affine.load %src[%i0]: memref<100xf32>
    affine.store %r1, %m[%i0] : memref<100xf32>
  }
  affine.for %i2 = 0 to 100 step 2 {
    %r2 = affine.load %m[%i2] : memref<100xf32>
    affine.store %r2, %out[%i2] : memref<100xf32>
  }
  return
}

// Check if the fusion did take place as well as that the source loop was
// not removed. To check if fusion took place, the read instruction from the
// original source loop is checked to be in the fused loop.
//
// PRODUCER-CONSUMER:        affine.for %[[idx_0:.*]] = 0 to 100 {
// PRODUCER-CONSUMER-NEXT:     %[[result_0:.*]] = affine.load %[[arr1:.*]][%[[idx_0]]] : memref<100xf32>
// PRODUCER-CONSUMER-NEXT:     affine.store %[[result_0]], %{{.*}}[%[[idx_0]]] : memref<100xf32>
// PRODUCER-CONSUMER-NEXT:   }
// PRODUCER-CONSUMER:        affine.for %[[idx_1:.*]] = 0 to 100 step 2 {
// PRODUCER-CONSUMER:          affine.load %[[arr1]][%[[idx_1]]] : memref<100xf32>
// PRODUCER-CONSUMER:        }
// PRODUCER-CONSUMER:        return

// -----

// SIBLING-MAXIMAL-LABEL:   func @reduce_add_non_maximal_f32_f32(
func.func @reduce_add_non_maximal_f32_f32(%arg0: memref<64x64xf32, 1>, %arg1 : memref<1x64xf32, 1>, %arg2 : memref<1x64xf32, 1>) {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    // This nest writes to %arg1 but can be eliminated post sibling fusion.
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 64 {
        %accum = affine.for %arg5 = 0 to 64 iter_args (%prevAccum = %cst_0) -> f32 {
          %4 = affine.load %arg0[%arg5, %arg4] : memref<64x64xf32, 1>
          %5 = arith.addf %prevAccum, %4 : f32
          affine.yield %5 : f32
        }
        %accum_dbl = arith.addf %accum, %accum : f32
        affine.store %accum_dbl, %arg1[%arg3, %arg4] : memref<1x64xf32, 1>
      }
    }
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 64 {
        // Following loop  trip count does not match the corresponding source trip count.
        %accum = affine.for %arg5 = 0 to 32 iter_args (%prevAccum = %cst_1) -> f32 {
          %4 = affine.load %arg0[%arg5, %arg4] : memref<64x64xf32, 1>
          %5 = arith.mulf %prevAccum, %4 : f32
          affine.yield %5 : f32
        }
        %accum_sqr = arith.mulf %accum, %accum : f32
        affine.store %accum_sqr, %arg2[%arg3, %arg4] : memref<1x64xf32, 1>
      }
    }
    return
}
// Test checks the loop structure is preserved after sibling fusion
// since the destination loop and source loop trip counts do not
// match.
// SIBLING-MAXIMAL:        %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
// SIBLING-MAXIMAL-NEXT:   %[[cst_1:.*]] = arith.constant 1.000000e+00 : f32
// SIBLING-MAXIMAL-NEXT:   affine.for %{{.*}} = 0 to 1 {
// SIBLING-MAXIMAL-NEXT:     affine.for %{{.*}} = 0 to 64 {
// SIBLING-MAXIMAL-NEXT:       affine.for %{{.*}} = 0 to 32 iter_args(%{{.*}} = %[[cst_1]]) -> (f32) {
// SIBLING-MAXIMAL-NEXT:       affine.for %{{.*}} = 0 to 64 iter_args(%{{.*}} = %[[cst_0]]) -> (f32) {

// -----

// SIBLING-MAXIMAL-LABEL: func @sibling_load_only
func.func @sibling_load_only(%arg0: memref<10xf32>) {
  affine.for %arg1 = 0 to 10 {
    %0 = affine.load %arg0[%arg1] : memref<10xf32>
  }
  affine.for %arg1 = 0 to 10 {
    %0 = affine.load %arg0[%arg1] : memref<10xf32>
  }
  // SIBLING-MAXIMAL-NEXT: affine.for
  // SIBLING-MAXIMAL-NEXT:   affine.load
  // SIBLING-MAXIMAL-NEXT:   affine.load
  return
}

// -----

// PRODUCER-CONSUMER-LABEL: func @fusion_for_multiple_blocks() {
func.func @fusion_for_multiple_blocks() {
^bb0:
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }
  // PRODUCER-CONSUMER:      affine.for %{{.*}} = 0 to 10 {
  // PRODUCER-CONSUMER-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // PRODUCER-CONSUMER-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // PRODUCER-CONSUMER-NEXT: }
  cf.br ^bb1
^bb1:
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }
  // PRODUCER-CONSUMER:      affine.for %{{.*}} = 0 to 10 {
  // PRODUCER-CONSUMER-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // PRODUCER-CONSUMER-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // PRODUCER-CONSUMER-NEXT: }
  return
}

// -----

// PRODUCER-CONSUMER-LABEL: @fuse_higher_dim_nest_into_lower_dim_nest
func.func @fuse_higher_dim_nest_into_lower_dim_nest() {
  %A = memref.alloc() : memref<8x12x128x64xf32>
  %B = memref.alloc() : memref<8x128x12x64xf32>
  affine.for %arg205 = 0 to 8 {
    affine.for %arg206 = 0 to 128 {
      affine.for %arg207 = 0 to 12 {
        affine.for %arg208 = 0 to 64 {
          %a = affine.load %A[%arg205, %arg207, %arg206, %arg208] : memref<8x12x128x64xf32>
          affine.store %a, %B[%arg205, %arg206, %arg207, %arg208] : memref<8x128x12x64xf32>
        }
      }
    }
  }
  %C = memref.alloc() : memref<8x128x768xf16>
  affine.for %arg205 = 0 to 8 {
    affine.for %arg206 = 0 to 128 {
      affine.for %arg207 = 0 to 768 {
        %b = affine.load %B[%arg205, %arg206, %arg207 floordiv 64, %arg207 mod 64] : memref<8x128x12x64xf32>
        %c = arith.truncf %b : f32 to f16
        affine.store %c, %C[%arg205, %arg206, %arg207] : memref<8x128x768xf16>
      }
    }
  }

  // Check that fusion happens into the innermost loop of the consumer.
  // PRODUCER-CONSUMER:      affine.for
  // PRODUCER-CONSUMER-NEXT:   affine.for %{{.*}} = 0 to 128
  // PRODUCER-CONSUMER-NEXT:     affine.for %{{.*}} = 0 to 768
  // PRODUCER-CONSUMER-NOT:  affine.for
  // PRODUCER-CONSUMER:      return
  return
}

// -----

// Basic test to ensure fusion works inside other func ops like spirv.func.

#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  // SPIRV-LABEL: func @test_avgpool2d_pad_right
  spirv.func @test_avgpool2d_pad_right(%arg0: !spirv.array<8192 x f32>) -> !spirv.array<8192 x f32> "None" {
    %cst_f32 = spirv.Constant 0.000000e+00 : f32
    %0 = builtin.unrealized_conversion_cast %arg0 : !spirv.array<8192 x f32> to tensor<1x32x32x8xf32>
    %padded = tensor.pad %0 low[0, 4, 4, 0] high[0, 4, 8193, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_f32 : f32
    } : tensor<1x32x32x8xf32> to tensor<1x40x8229x8xf32>
    %1 = bufferization.to_memref %padded : tensor<1x40x8229x8xf32> to memref<1x40x8229x8xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x8xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 32 {
        affine.for %arg3 = 0 to 32 {
          affine.for %arg4 = 0 to 8 {
            affine.for %arg5 = 0 to 1 {
              affine.for %arg6 = 0 to 1 {
                %4 = affine.apply #map(%arg2, %arg5)
                %5 = affine.apply #map(%arg3, %arg6)
                %6 = affine.load %1[%arg1, %4, %5, %arg4] : memref<1x40x8229x8xf32>
                %7 = affine.load %alloc_0[%arg1, %arg2, %arg3, %arg4] : memref<1x32x32x8xf32>
                %8 = arith.addf %7, %6 : f32
                affine.store %8, %alloc_0[%arg1, %arg2, %arg3, %arg4] : memref<1x32x32x8xf32>
              }
            }
          }
        }
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x8xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 32 {
        affine.for %arg3 = 0 to 32 {
          affine.for %arg4 = 0 to 8 {
            %4 = affine.load %alloc_0[%arg1, %arg2, %arg3, %arg4] : memref<1x32x32x8xf32>
          }
        }
      }
    }
    // Test fusion.
    // SPIRV:      affine.for %{{.*}} = 0 to 1 {
    // SPIRV-NEXT:   affine.for %{{.*}} = 0 to 32 {
    // SPIRV-NEXT:     affine.for %{{.*}} = 0 to 32 {
    // SPIRV-NEXT:       affine.for %{{.*}} = 0 to 8 {
    // SPIRV-NOT:       affine.for %{{.*}}

    // SPIRV:       ReturnValue
    %2 = bufferization.to_tensor %alloc_1 : memref<1x32x32x8xf32> to tensor<1x32x32x8xf32>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x32x8xf32> to !spirv.array<8192 x f32>
    spirv.ReturnValue %3 : !spirv.array<8192 x f32>
  }
}

// -----

// PRODUCER-CONSUMER-LABEL: func @same_memref_load_store
func.func @same_memref_load_store(%producer : memref<32xf32>, %consumer: memref<16xf32>){
  %cst = arith.constant 2.000000e+00 : f32
  // Source isn't removed.
  // PRODUCER-CONSUMER: affine.for %{{.*}} = 0 to 32
  affine.for %arg3 = 0 to 32 {
    %0 = affine.load %producer[%arg3] : memref<32xf32>
    %2 = arith.mulf %0, %cst : f32
    affine.store %2, %producer[%arg3] : memref<32xf32>
  }
  affine.for %arg3 = 0 to 16 {
    %0 = affine.load %producer[%arg3] : memref<32xf32>
    %2 = arith.addf %0, %cst : f32
    affine.store %2, %consumer[%arg3] : memref<16xf32>
  }
  // Fused nest.
  // PRODUCER-CONSUMER:      affine.for %{{.*}} = 0 to 16
  // PRODUCER-CONSUMER-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<32xf32>
  // PRODUCER-CONSUMER-NEXT:   arith.mulf
  // PRODUCER-CONSUMER-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // PRODUCER-CONSUMER-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // PRODUCER-CONSUMER-NEXT:   arith.addf
  // PRODUCER-CONSUMER-NEXT:   affine.store
  // PRODUCER-CONSUMER-NEXT: }
  return
}

// -----

// PRODUCER-CONSUMER-LABEL: func @same_memref_load_multiple_stores
// ALL-LABEL: func @same_memref_load_multiple_stores
func.func @same_memref_load_multiple_stores(%producer : memref<32xf32>, %producer_2 : memref<32xf32>, %consumer: memref<16xf32>){
  %cst = arith.constant 2.000000e+00 : f32
  // Ensure that source isn't removed during both producer-consumer fusion and
  // sibling fusion.
  // PRODUCER-CONSUMER: affine.for %{{.*}} = 0 to 32
  // ALL: affine.for %{{.*}} = 0 to 32
  affine.for %arg3 = 0 to 32 {
    %0 = affine.load %producer[%arg3] : memref<32xf32>
    %2 = arith.mulf %0, %cst : f32
    affine.store %2, %producer[%arg3] : memref<32xf32>
    affine.store %2, %producer_2[%arg3] : memref<32xf32>
  }
  affine.for %arg3 = 0 to 16 {
    %0 = affine.load %producer[%arg3] : memref<32xf32>
    %1 = affine.load %producer_2[%arg3] : memref<32xf32>
    %2 = arith.addf %0, %1 : f32
    affine.store %2, %consumer[%arg3] : memref<16xf32>
  }
  // Fused nest.
  // PRODUCER-CONSUMER:      affine.for %{{.*}} = 0 to 16
  // PRODUCER-CONSUMER-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<32xf32>
  // PRODUCER-CONSUMER-NEXT:   arith.mulf
  // PRODUCER-CONSUMER-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // PRODUCER-CONSUMER-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // PRODUCER-CONSUMER-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // PRODUCER-CONSUMER-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // PRODUCER-CONSUMER-NEXT:   arith.addf
  // PRODUCER-CONSUMER-NEXT:   affine.store
  // PRODUCER-CONSUMER-NEXT: }
  // ALL:     affine.for %{{.*}} = 0 to 16
  // ALL:       mulf
  // ALL:       addf
  return
}

// -----

#map = affine_map<()[s0] -> (s0 + 5)>
#map1 = affine_map<()[s0] -> (s0 + 17)>

// Test with non-int/float memref types.

// PRODUCER-CONSUMER-MAXIMAL-LABEL: func @memref_index_type
func.func @memref_index_type() {
  %0 = llvm.mlir.constant(2 : index) : i64
  %2 = llvm.mlir.constant(0 : index) : i64
  %3 = builtin.unrealized_conversion_cast %2 : i64 to index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x18xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<3xf32>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
  affine.for %arg3 = 0 to 3 {
    %4 = affine.load %alloc_2[%arg3] : memref<3xindex>
    %5 = builtin.unrealized_conversion_cast %4 : index to i64
    %6 = llvm.sub %0, %5 : i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to index
    affine.store %7, %alloc_2[%arg3] : memref<3xindex>
  }
  affine.for %arg3 = 0 to 3 {
    %4 = affine.load %alloc_2[%arg3] : memref<3xindex>
    %5 = affine.apply #map()[%4]
    %6 = affine.apply #map1()[%3]
    %7 = memref.load %alloc[%5, %6] : memref<8x18xf32>
    affine.store %7, %alloc_1[%arg3] : memref<3xf32>
  }
  // Expect fusion.
  // PRODUCER-CONSUMER-MAXIMAL: affine.for
  // PRODUCER-CONSUMER-MAXIMAL-NOT: affine.for
  // PRODUCER-CONSUMER-MAXIMAL: return
  return
}

// -----

#map = affine_map<(d0) -> (d0)>
#map1 =affine_map<(d0) -> (d0 + 1)>

// Test non-integer memory spaces.

// PRODUCER-CONSUMER-LABEL: func @non_int_memory_space
func.func @non_int_memory_space() {
  %alloc = memref.alloc() : memref<256x8xf32, #spirv.storage_class<StorageBuffer>>
  affine.for %arg0 = 0 to 64 {
    affine.for %arg1 = 0 to 8 {
      %0 = affine.apply #map(%arg1)
      %1 = affine.load %alloc[%arg0, %0] : memref<256x8xf32, #spirv.storage_class<StorageBuffer>>
      affine.store %1, %alloc[%arg0, %arg1] : memref<256x8xf32, #spirv.storage_class<StorageBuffer>>
    }
  }
  affine.for %arg0 = 16 to 32 {
    affine.for %arg1 = 0 to 8 {
      %0 = affine.apply #map(%arg1)
      %1 = affine.load %alloc[%arg0, %0] : memref<256x8xf32, #spirv.storage_class<StorageBuffer>>
      affine.store %1, %alloc[%arg0, %arg1] : memref<256x8xf32, #spirv.storage_class<StorageBuffer>>
    }
  }
  // Fused nest.
  // PRODUCER-CONSUMER-NEXT: memref.alloc()
  // PRODUCER-CONSUMER-NEXT: memref.alloc()
  // PRODUCER-CONSUMER-NEXT: affine.for %{{.*}} = 16 to 32
  // PRODUCER-CONSUMER-NEXT:   affine.for %{{.*}} = 0 to 8
  return
}

// -----

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>

// Exercises fix for crash reported at https://github.com/llvm/llvm-project/issues/119525

// No fusion of  producer into consumer happens here as the slice is determined
// to be invalid. This is a limitation and it is possible to compute a slice
// (reduction along %arg4) and fuse.

// PRODUCER-CONSUMER-LABEL: func @slice_compute_check
func.func @slice_compute_check(%arg0: memref<1x8x26xi32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<1x8x26xi32, strided<[?, ?, ?], offset: ?>>, %arg2: memref<1x8x26xi32, strided<[?, ?, ?], offset: ?>>) {
  %alloc_14 = memref.alloc() : memref<1x8x26xi32>
  %alloc_15 = memref.alloc() : memref<1x26xi32>
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 8 {
      affine.for %arg5 = 0 to 26 {
        affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
          affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
            affine.for %arg8 = #map(%arg5) to #map1(%arg5) {
              %61 = affine.load %alloc_14[%arg6, %arg7, %arg8] : memref<1x8x26xi32>
              %62 = affine.load %alloc_15[%arg6, %arg8] : memref<1x26xi32>
              %63 = llvm.intr.smin(%61, %62) : (i32, i32) -> i32
              affine.store %63, %alloc_15[%arg6, %arg8] : memref<1x26xi32>
            }
          }
        }
      }
    }
  }
  affine.for %arg3 = 0 to 26 {
    %61 = affine.load %alloc_15[0, %arg3] : memref<1x26xi32>
  }
  memref.dealloc %alloc_15 : memref<1x26xi32>
  memref.dealloc %alloc_14 : memref<1x8x26xi32>
  return
}

// -----

// Exercises fix for crash reported at https://github.com/llvm/llvm-project/issues/108374

// No fusion of  producer into consumer happens here. The slice will not be
// valid as the producer doesn't supply to all of the consumer.

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
// PRODUCER-CONSUMER-LABEL: func @test_add_slice_bounds
func.func @test_add_slice_bounds() {
  %alloc = memref.alloc() : memref<10xf32>
  %cst = arith.constant 0.619152 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = #map(%arg0) to #map1(%arg0) {
      affine.store %cst, %alloc[%arg1] : memref<10xf32>
    }
  }
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = #map(%arg0) to #map1(%arg0) {
        affine.for %arg3 = #map(%arg1) to #map1(%arg1) {
          %0 = affine.apply #map1(%arg3)
          %1 = affine.load %alloc[%0] : memref<10xf32>
        }
      }
    }
  }
  return
}

// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm="enable-x86vector" -convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | \
// RUN: mlir-translate  --mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="avx512bw,avx512vp2intersect" --dlopen=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// This test shows how to implement a sparse vector-vector dot product with
// AVX512. It uses vp2intersect, mask.compress and vector.contract to compute
// the dot product of two sparse HW vectors of 8 float64 elements ("segment").
// Each sparse vector is represented by an index memref (A or C) and by a data
// memref (B or D), containing M or N elements.
//
// There are four different implementations:
// * `memref_dot_simple`: Simple O(N*M) implementation with two for loops.
// * `memref_dot_optimized`: An optimized O(N*M) version of the previous
//   implementation, where the second for loop skips over some elements.
// * `memref_dot_while`: An optimized O(N+M) implementation that utilizes a
//   single while loop, coiterating over both vectors.
// * `memref_dot_while_branchless`: An optimized O(N+M) implementation that
//   consists of a single while loop and has no branches within the loop.
//
// Output of llvm-mca:
// https://gist.github.com/matthias-springer/72e7ee1b3c467e7aefb6e1fd862e4841

#contraction_accesses = [
 affine_map<(i) -> (i)>,
 affine_map<(i) -> (i)>,
 affine_map<(i) -> ()>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["reduction"]
}

// Sparse vector dot product of two vectors.
func.func @vector_dot(%v_A : vector<8xi64>, %v_B : vector<8xf64>,
                 %v_C : vector<8xi64>, %v_D : vector<8xf64>) -> f64 {
  // Compute intersection of indices.
  %k0, %k1 = x86vector.avx512.vp2intersect %v_A, %v_C : vector<8xi64>

  // Filter out values without match and compress vector.
  %p0 = x86vector.avx512.mask.compress %k0, %v_B : vector<8xf64>
  %p1 = x86vector.avx512.mask.compress %k1, %v_D : vector<8xf64>

  // Dense vector dot product.
  %acc = arith.constant 0.0 : f64
  %r = vector.contract #contraction_trait %p0, %p1, %acc
      : vector<8xf64>, vector<8xf64> into f64

  return %r : f64
}

// Fill input memrefs will all zeros, so that they can be used with arbitrary
// input sizes up to 128 elements per sparse vector.
func.func @init_input(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                 %m_C : memref<?xi64>, %m_D : memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %v_data = arith.constant dense<0.0> : vector<128xf64>
  %v_index = arith.constant dense<9223372036854775807> : vector<128xi64>

  vector.transfer_write %v_index, %m_A[%c0] : vector<128xi64>, memref<?xi64>
  vector.transfer_write %v_data, %m_B[%c0] : vector<128xf64>, memref<?xf64>
  vector.transfer_write %v_index, %m_C[%c0] : vector<128xi64>, memref<?xi64>
  vector.transfer_write %v_data, %m_D[%c0] : vector<128xf64>, memref<?xf64>

  return
}

func.func @fill_input_1(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                   %m_C : memref<?xi64>, %m_D : memref<?xf64>)
    -> (index, index){
  func.call @init_input(%m_A, %m_B, %m_C, %m_D)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>) -> ()

  %c0 = arith.constant 0 : index

  %v_A = arith.constant dense<[0,  1,  10, 12, 13, 17, 18, 21,
                            51, 52, 57, 61, 62, 82, 98, 99]> : vector<16xi64>
  %v_B = arith.constant dense<[1., 5., 8., 3., 2., 1., 0., 9.,
                            6., 7., 7., 3., 5., 2., 9., 1.]> : vector<16xf64>
  %v_C = arith.constant dense<[1,  2,  5,  10, 11, 12, 47, 48,
                            67, 68, 69, 70, 71, 72, 77, 78,
                            79, 82, 83, 84, 85, 90, 91, 98]> : vector<24xi64>
  %v_D = arith.constant dense<[1., 5., 8., 3., 2., 1., 2., 9.,
                            6., 7., 7., 3., 5., 2., 9., 1.,
                            2., 9., 8., 7., 2., 0., 0., 4.]> : vector<24xf64>

  vector.transfer_write %v_A, %m_A[%c0] : vector<16xi64>, memref<?xi64>
  vector.transfer_write %v_B, %m_B[%c0] : vector<16xf64>, memref<?xf64>
  vector.transfer_write %v_C, %m_C[%c0] : vector<24xi64>, memref<?xi64>
  vector.transfer_write %v_D, %m_D[%c0] : vector<24xf64>, memref<?xf64>

  %M = arith.constant 16 : index
  %N = arith.constant 24 : index

  return %M, %N : index, index
}

func.func @fill_input_2(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                   %m_C : memref<?xi64>, %m_D : memref<?xf64>)
    -> (index, index){
  func.call @init_input(%m_A, %m_B, %m_C, %m_D)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>) -> ()

  %c0 = arith.constant 0 : index

  %v_A = arith.constant dense<[0,  1,  3,  5,  6,  7,  8,  9,
                            51, 52, 57, 61, 62, 63, 65, 66]> : vector<16xi64>
  %v_B = arith.constant dense<[1., 5., 8., 3., 2., 1., 2., 9.,
                            6., 7., 7., 3., 5., 2., 9., 1.]> : vector<16xf64>
  %v_C = arith.constant dense<[6,  7,  11, 12, 15, 17, 19, 21,
                            30, 31, 33, 34, 37, 39, 40, 41,
                            42, 44, 45, 46, 47, 48, 49, 50,
                            62, 63, 64, 65, 66, 67, 68, 69,
                            70, 77, 78, 79, 81, 82, 89, 99]> : vector<40xi64>
  %v_D = arith.constant dense<[1., 5., 8., 3., 2., 1., 2., 9.,
                            6., 7., 7., 3., 5., 2., 9., 1.,
                            2., 9., 8., 7., 2., 1., 2., 4.,
                            4., 5., 8., 8., 2., 3., 5., 1.,
                            8., 6., 6., 4., 3., 8., 9., 2.]> : vector<40xf64>

  vector.transfer_write %v_A, %m_A[%c0] : vector<16xi64>, memref<?xi64>
  vector.transfer_write %v_B, %m_B[%c0] : vector<16xf64>, memref<?xf64>
  vector.transfer_write %v_C, %m_C[%c0] : vector<40xi64>, memref<?xi64>
  vector.transfer_write %v_D, %m_D[%c0] : vector<40xf64>, memref<?xf64>

  %M = arith.constant 16 : index
  %N = arith.constant 40 : index

  return %M, %N : index, index
}

// Simple vector dot product implementation: Intersect every segment of size 8
// in (%m_A, %m_B) with every segment of size 8 in (%m_C, %m_D).
func.func @memref_dot_simple(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                        %m_C : memref<?xi64>, %m_D : memref<?xf64>,
                        %M : index, %N : index)
    -> f64 {
  // Helper constants for loops.
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index

  %data_zero = arith.constant 0.0 : f64
  %index_padding = arith.constant 9223372036854775807 : i64

  // Notation: %sum is the current (partial) aggregated dot product sum.

  %r0 = scf.for %a = %c0 to %M step %c8
      iter_args(%sum0 = %data_zero) -> (f64) {
    %v_A = vector.transfer_read %m_A[%a], %index_padding
        : memref<?xi64>, vector<8xi64>
    %v_B = vector.transfer_read %m_B[%a], %data_zero
        : memref<?xf64>, vector<8xf64>

    %r1 = scf.for %b = %c0 to %N step %c8
        iter_args(%sum1 = %sum0) -> (f64) {
      %v_C = vector.transfer_read %m_C[%b], %index_padding
          : memref<?xi64>, vector<8xi64>
      %v_D = vector.transfer_read %m_D[%b], %data_zero
          : memref<?xf64>, vector<8xf64>

      %subresult = func.call @vector_dot(%v_A, %v_B, %v_C, %v_D)
          : (vector<8xi64>, vector<8xf64>, vector<8xi64>, vector<8xf64>) -> f64
      %r2 = arith.addf %sum1, %subresult : f64
      scf.yield %r2 : f64
    }

    scf.yield %r1 : f64
  }

  return %r0 : f64
}

// Optimized vector dot product implementation: Taking advantage of the fact
// that indices in %m_A and %m_C are sorted ascendingly, skip over segments
// in (%m_C, %m_D) that are know to have no intersection with the current
// segment from (%m_A, %m_B).
func.func @memref_dot_optimized(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                           %m_C : memref<?xi64>, %m_D : memref<?xf64>,
                           %M : index, %N : index)
    -> f64 {
  // Helper constants for loops.
  %c0 = arith.constant 0 : index
  %i0 = arith.constant 0 : i32
  %i7 = arith.constant 7 : i32
  %c8 = arith.constant 8 : index

  %data_zero = arith.constant 0.0 : f64
  %index_padding = arith.constant 9223372036854775807 : i64

  // Notation: %sum is the current (partial) aggregated dot product sum.
  // %j_start is the value from which the inner for loop starts iterating. This
  // value keeps increasing if earlier segments of (%m_C, %m_D) are known to
  // be no longer needed.

  %r0, %t0 = scf.for %a = %c0 to %M step %c8
      iter_args(%sum0 = %data_zero, %b_start0 = %c0) -> (f64, index) {
    %v_A = vector.transfer_read %m_A[%a], %index_padding
        : memref<?xi64>, vector<8xi64>
    %segA_min = vector.extractelement %v_A[%i0 : i32] : vector<8xi64>

    %r1, %next_b_start0 = scf.for %b = %b_start0 to %N step %c8
        iter_args(%sum1 = %sum0, %b_start1 = %b_start0) -> (f64, index) {
      %v_C = vector.transfer_read %m_C[%b], %index_padding
          : memref<?xi64>, vector<8xi64>
      %segB_max = vector.extractelement %v_C[%i7 : i32] : vector<8xi64>
      %seg1_done = arith.cmpi "slt", %segB_max, %segA_min : i64

      %r2, %next_b_start1 = scf.if %seg1_done -> (f64, index) {
        // %v_C segment is done, no need to examine this one again (ever).
        %next_b_start2 = arith.addi %b_start1, %c8 : index
        scf.yield %sum1, %next_b_start2 : f64, index
      } else {
        %v_B = vector.transfer_read %m_B[%a], %data_zero
            : memref<?xf64>, vector<8xf64>
        %v_D = vector.transfer_read %m_D[%b], %data_zero
            : memref<?xf64>, vector<8xf64>

        %subresult = func.call @vector_dot(%v_A, %v_B, %v_C, %v_D)
            : (vector<8xi64>, vector<8xf64>, vector<8xi64>, vector<8xf64>)
                -> f64
        %r3 = arith.addf %sum1, %subresult : f64
        scf.yield %r3, %b_start1 : f64, index
      }

      scf.yield %r2, %next_b_start1 : f64, index
    }

    scf.yield %r1, %next_b_start0 : f64, index
  }

  return %r0 : f64
}

// Vector dot product with a while loop. Implemented as follows:
//
// r = 0.0, a = 0, b = 0
// while (a < M && b < N) {
//   segA = A[a:a+8], segB = B[b:b+8]
//   if   (segB[7] < segA[0]) b += 8
//   elif (segA[7] < segB[0]) a += 8
//   else {
//     r += vector_dot(...)
//     if   (segA[7] < segB[7]) a += 8
//     elif (segB[7] < segA[7]) b += 8
//     else                     a += 8, b += 8
//   }
// }
func.func @memref_dot_while(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                       %m_C : memref<?xi64>, %m_D : memref<?xf64>,
                       %M : index, %N : index)
    -> f64 {
  // Helper constants for loops.
  %c0 = arith.constant 0 : index
  %i0 = arith.constant 0 : i32
  %i7 = arith.constant 7 : i32
  %c8 = arith.constant 8 : index

  %data_zero = arith.constant 0.0 : f64
  %index_padding = arith.constant 9223372036854775807 : i64

  %r0, %a0, %b0 = scf.while (%r1 = %data_zero, %a1 = %c0, %b1 = %c0)
      : (f64, index, index) -> (f64, index, index) {
    %cond_i = arith.cmpi "slt", %a1, %M : index
    %cond_j = arith.cmpi "slt", %b1, %N : index
    %cond = arith.andi %cond_i, %cond_j : i1
    scf.condition(%cond) %r1, %a1, %b1 : f64, index, index
  } do {
  ^bb0(%r1 : f64, %a1 : index, %b1 : index):
    // v_A, v_B, seg*_* could be part of the loop state to avoid a few
    // redundant reads.
    %v_A = vector.transfer_read %m_A[%a1], %index_padding
        : memref<?xi64>, vector<8xi64>
    %v_C = vector.transfer_read %m_C[%b1], %index_padding
        : memref<?xi64>, vector<8xi64>

    %segA_min = vector.extractelement %v_A[%i0 : i32] : vector<8xi64>
    %segA_max = vector.extractelement %v_A[%i7 : i32] : vector<8xi64>
    %segB_min = vector.extractelement %v_C[%i0 : i32] : vector<8xi64>
    %segB_max = vector.extractelement %v_C[%i7 : i32] : vector<8xi64>

    %seg1_done = arith.cmpi "slt", %segB_max, %segA_min : i64
    %r2, %a2, %b2 = scf.if %seg1_done -> (f64, index, index) {
      %b3 = arith.addi %b1, %c8 : index
      scf.yield %r1, %a1, %b3 : f64, index, index
    } else {
      %seg0_done = arith.cmpi "slt", %segA_max, %segB_min : i64
      %r4, %a4, %b4 = scf.if %seg0_done -> (f64, index, index) {
        %a5 = arith.addi %a1, %c8 : index
        scf.yield %r1, %a5, %b1 : f64, index, index
      } else {
        %v_B = vector.transfer_read %m_B[%a1], %data_zero
            : memref<?xf64>, vector<8xf64>
        %v_D = vector.transfer_read %m_D[%b1], %data_zero
            : memref<?xf64>, vector<8xf64>

        %subresult = func.call @vector_dot(%v_A, %v_B, %v_C, %v_D)
            : (vector<8xi64>, vector<8xf64>, vector<8xi64>, vector<8xf64>)
                -> f64
        %r6 = arith.addf %r1, %subresult : f64

        %incr_a = arith.cmpi "slt", %segA_max, %segB_max : i64
        %a6, %b6 = scf.if %incr_a -> (index, index) {
          %a7 = arith.addi %a1, %c8 : index
          scf.yield %a7, %b1 : index, index
        } else {
          %incr_b = arith.cmpi "slt", %segB_max, %segA_max : i64
          %a8, %b8 = scf.if %incr_b -> (index, index) {
            %b9 = arith.addi %b1, %c8 : index
            scf.yield %a1, %b9 : index, index
          } else {
            %a10 = arith.addi %a1, %c8 : index
            %b10 = arith.addi %b1, %c8 : index
            scf.yield %a10, %b10 : index, index
          }
          scf.yield %a8, %b8 : index, index
        }
        scf.yield %r6, %a6, %b6 : f64, index, index
      }
      scf.yield %r4, %a4, %b4 : f64, index, index
    }
    scf.yield %r2, %a2, %b2 : f64, index, index
  }

  return %r0 : f64
}

// Vector dot product with a while loop that has no branches (apart from the
// while loop itself). Implemented as follows:
//
// r = 0.0, a = 0, b = 0
// while (a < M && b < N) {
//   segA = A[a:a+8], segB = B[b:b+8]
//   r += vector_dot(...)
//   a += (segA[7] <= segB[7]) * 8
//   b += (segB[7] <= segA[7]) * 8
// }
func.func @memref_dot_while_branchless(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                                  %m_C : memref<?xi64>, %m_D : memref<?xf64>,
                                  %M : index, %N : index)
    -> f64 {
  // Helper constants for loops.
  %c0 = arith.constant 0 : index
  %i7 = arith.constant 7 : i32
  %c8 = arith.constant 8 : index

  %data_zero = arith.constant 0.0 : f64
  %index_padding = arith.constant 9223372036854775807 : i64

  %r0, %a0, %b0 = scf.while (%r1 = %data_zero, %a1 = %c0, %b1 = %c0)
      : (f64, index, index) -> (f64, index, index) {
    %cond_i = arith.cmpi "slt", %a1, %M : index
    %cond_j = arith.cmpi "slt", %b1, %N : index
    %cond = arith.andi %cond_i, %cond_j : i1
    scf.condition(%cond) %r1, %a1, %b1 : f64, index, index
  } do {
  ^bb0(%r1 : f64, %a1 : index, %b1 : index):
    // v_A, v_B, seg*_* could be part of the loop state to avoid a few
    // redundant reads.
    %v_A = vector.transfer_read %m_A[%a1], %index_padding
        : memref<?xi64>, vector<8xi64>
    %v_B = vector.transfer_read %m_B[%a1], %data_zero
        : memref<?xf64>, vector<8xf64>
    %v_C = vector.transfer_read %m_C[%b1], %index_padding
        : memref<?xi64>, vector<8xi64>
    %v_D = vector.transfer_read %m_D[%b1], %data_zero
        : memref<?xf64>, vector<8xf64>

    %subresult = func.call @vector_dot(%v_A, %v_B, %v_C, %v_D)
        : (vector<8xi64>, vector<8xf64>, vector<8xi64>, vector<8xf64>)
            -> f64
    %r2 = arith.addf %r1, %subresult : f64

    %segA_max = vector.extractelement %v_A[%i7 : i32] : vector<8xi64>
    %segB_max = vector.extractelement %v_C[%i7 : i32] : vector<8xi64>

    %cond_a = arith.cmpi "sle", %segA_max, %segB_max : i64
    %cond_a_i64 = arith.extui %cond_a : i1 to i64
    %cond_a_idx = arith.index_cast %cond_a_i64 : i64 to index
    %incr_a = arith.muli %cond_a_idx, %c8 : index
    %a2 = arith.addi %a1, %incr_a : index

    %cond_b = arith.cmpi "sle", %segB_max, %segA_max : i64
    %cond_b_i64 = arith.extui %cond_b : i1 to i64
    %cond_b_idx = arith.index_cast %cond_b_i64 : i64 to index
    %incr_b = arith.muli %cond_b_idx, %c8 : index
    %b2 = arith.addi %b1, %incr_b : index

    scf.yield %r2, %a2, %b2 : f64, index, index
  }

  return %r0 : f64
}

func.func @entry() -> i32 {
  // Initialize large buffers that can be used for multiple test cases of
  // different sizes.
  %b_A = memref.alloc() : memref<128xi64>
  %b_B = memref.alloc() : memref<128xf64>
  %b_C = memref.alloc() : memref<128xi64>
  %b_D = memref.alloc() : memref<128xf64>

  %m_A = memref.cast %b_A : memref<128xi64> to memref<?xi64>
  %m_B = memref.cast %b_B : memref<128xf64> to memref<?xf64>
  %m_C = memref.cast %b_C : memref<128xi64> to memref<?xi64>
  %m_D = memref.cast %b_D : memref<128xf64> to memref<?xf64>

  // --- Test case 1 ---.
  // M and N must be a multiple of 8 if smaller than 128.
  // (Because padding kicks in only for out-of-bounds accesses.)
  %M1, %N1 = func.call @fill_input_1(%m_A, %m_B, %m_C, %m_D)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>)
          -> (index, index)

  %r0 = func.call @memref_dot_simple(%m_A, %m_B, %m_C, %m_D, %M1, %N1)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r0 : f64
  // CHECK: 86

  %r1 = func.call @memref_dot_optimized(%m_A, %m_B, %m_C, %m_D, %M1, %N1)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r1 : f64
  // CHECK: 86

  %r2 = func.call @memref_dot_while(%m_A, %m_B, %m_C, %m_D, %M1, %N1)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r2 : f64
  // CHECK: 86

  %r6 = func.call @memref_dot_while_branchless(%m_A, %m_B, %m_C, %m_D, %M1, %N1)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r6 : f64
  // CHECK: 86

  // --- Test case 2 ---.
  // M and N must be a multiple of 8 if smaller than 128.
  // (Because padding kicks in only for out-of-bounds accesses.)
  %M2, %N2 = func.call @fill_input_2(%m_A, %m_B, %m_C, %m_D)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>)
          -> (index, index)

  %r3 = func.call @memref_dot_simple(%m_A, %m_B, %m_C, %m_D, %M2, %N2)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r3 : f64
  // CHECK: 111

  %r4 = func.call @memref_dot_optimized(%m_A, %m_B, %m_C, %m_D, %M2, %N2)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r4 : f64
  // CHECK: 111

  %r5 = func.call @memref_dot_while(%m_A, %m_B, %m_C, %m_D, %M2, %N2)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r5 : f64
  // CHECK: 111

  %r7 = func.call @memref_dot_while_branchless(%m_A, %m_B, %m_C, %m_D, %M2, %N2)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r7 : f64
  // CHECK: 111

  // Release all resources.
  memref.dealloc %b_A : memref<128xi64>
  memref.dealloc %b_B : memref<128xf64>
  memref.dealloc %b_C : memref<128xi64>
  memref.dealloc %b_D : memref<128xf64>

  %r = arith.constant 0 : i32
  return %r : i32
}

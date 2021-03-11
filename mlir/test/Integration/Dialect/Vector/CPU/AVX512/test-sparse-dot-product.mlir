// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm="enable-avx512" -convert-std-to-llvm  | \
// RUN: mlir-translate  --mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="avx512bw,avx512vp2intersect" --dlopen=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// This test shows how to implement a sparse vector-vector dot product with
// AVX512. It uses vp2intersect, mask.compress and vector.contract to compute
// the dot product of two sparse HW vectors of 8 float64 elements ("segment").
// Each sparse vector is represented by an index memref (A or C) and by a data
// memref (B or D), containing M or N elements.
//
// There are two implementations:
// * `memref_dot_simple`: Simple O(N*M) implementation with two for loops.
// * `memref_dot_optimized`: An optimized O(N*M) version of the previous
//   implementation, where the second for loop skips over some elements.

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
func @vector_dot(%v_A : vector<8xi64>, %v_B : vector<8xf64>,
                 %v_C : vector<8xi64>, %v_D : vector<8xf64>) -> f64 {
  // Compute intersection of indices.
  %k0, %k1 = avx512.vp2intersect %v_A, %v_C : vector<8xi64>

  // Filter out values without match and compress vector.
  %p0 = avx512.mask.compress %k0, %v_B : vector<8xf64>
  %p1 = avx512.mask.compress %k1, %v_D : vector<8xf64>

  // Dense vector dot product.
  %acc = std.constant 0.0 : f64
  %r = vector.contract #contraction_trait %p0, %p1, %acc
      : vector<8xf64>, vector<8xf64> into f64

  return %r : f64
}

// Fill input memrefs will all zeros, so that they can be used with arbitrary
// input sizes up to 128 elements per sparse vector.
func @init_input(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                 %m_C : memref<?xi64>, %m_D : memref<?xf64>) {
  %c0 = constant 0 : index
  %v_data = constant dense<0.0> : vector<128xf64>
  %v_index = constant dense<9223372036854775807> : vector<128xi64>

  vector.transfer_write %v_index, %m_A[%c0] : vector<128xi64>, memref<?xi64>
  vector.transfer_write %v_data, %m_B[%c0] : vector<128xf64>, memref<?xf64>
  vector.transfer_write %v_index, %m_C[%c0] : vector<128xi64>, memref<?xi64>
  vector.transfer_write %v_data, %m_D[%c0] : vector<128xf64>, memref<?xf64>

  return
}

func @fill_input_1(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                   %m_C : memref<?xi64>, %m_D : memref<?xf64>)
    -> (index, index){
  call @init_input(%m_A, %m_B, %m_C, %m_D)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>) -> ()

  %c0 = constant 0 : index

  %v_A = std.constant dense<[0,  1,  10, 12, 13, 17, 18, 21,
                             51, 52, 57, 61, 62, 82, 98, 99]> : vector<16xi64>
  %v_B = std.constant dense<[1., 5., 8., 3., 2., 1., 0., 9.,
                             6., 7., 7., 3., 5., 2., 9., 1.]> : vector<16xf64>
  %v_C = std.constant dense<[1,  2,  5,  10, 11, 12, 47, 48,
                             67, 68, 69, 70, 71, 72, 77, 78,
                             79, 82, 83, 84, 85, 90, 91, 98]> : vector<24xi64>
  %v_D = std.constant dense<[1., 5., 8., 3., 2., 1., 2., 9.,
                             6., 7., 7., 3., 5., 2., 9., 1.,
                             2., 9., 8., 7., 2., 0., 0., 4.]> : vector<24xf64>

  vector.transfer_write %v_A, %m_A[%c0] : vector<16xi64>, memref<?xi64>
  vector.transfer_write %v_B, %m_B[%c0] : vector<16xf64>, memref<?xf64>
  vector.transfer_write %v_C, %m_C[%c0] : vector<24xi64>, memref<?xi64>
  vector.transfer_write %v_D, %m_D[%c0] : vector<24xf64>, memref<?xf64>

  %M = std.constant 16 : index
  %N = std.constant 24 : index

  return %M, %N : index, index
}

func @fill_input_2(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                   %m_C : memref<?xi64>, %m_D : memref<?xf64>)
    -> (index, index){
  call @init_input(%m_A, %m_B, %m_C, %m_D)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>) -> ()

  %c0 = constant 0 : index

  %v_A = std.constant dense<[0,  1,  3,  5,  6,  7,  8,  9,
                             51, 52, 57, 61, 62, 63, 65, 66]> : vector<16xi64>
  %v_B = std.constant dense<[1., 5., 8., 3., 2., 1., 2., 9.,
                             6., 7., 7., 3., 5., 2., 9., 1.]> : vector<16xf64>
  %v_C = std.constant dense<[6,  7,  11, 12, 15, 17, 19, 21,
                             30, 31, 33, 34, 37, 39, 40, 41,
                             42, 44, 45, 46, 47, 48, 49, 50,
                             62, 63, 64, 65, 66, 67, 68, 69,
                             70, 77, 78, 79, 81, 82, 89, 99]> : vector<40xi64>
  %v_D = std.constant dense<[1., 5., 8., 3., 2., 1., 2., 9.,
                             6., 7., 7., 3., 5., 2., 9., 1.,
                             2., 9., 8., 7., 2., 1., 2., 4.,
                             4., 5., 8., 8., 2., 3., 5., 1.,
                             8., 6., 6., 4., 3., 8., 9., 2.]> : vector<40xf64>

  vector.transfer_write %v_A, %m_A[%c0] : vector<16xi64>, memref<?xi64>
  vector.transfer_write %v_B, %m_B[%c0] : vector<16xf64>, memref<?xf64>
  vector.transfer_write %v_C, %m_C[%c0] : vector<40xi64>, memref<?xi64>
  vector.transfer_write %v_D, %m_D[%c0] : vector<40xf64>, memref<?xf64>

  %M = std.constant 16 : index
  %N = std.constant 40 : index

  return %M, %N : index, index
}

// Simple vector dot product implementation: Intersect every segment of size 8
// in (%m_A, %m_B) with every segment of size 8 in (%m_C, %m_D).
func @memref_dot_simple(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                        %m_C : memref<?xi64>, %m_D : memref<?xf64>,
                        %M : index, %N : index)
    -> f64 {
  // Helper constants for loops.
  %c0 = constant 0 : index
  %c8 = constant 8 : index

  %data_zero = constant 0.0 : f64
  %index_padding = constant 9223372036854775807 : i64

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

      %subresult = call @vector_dot(%v_A, %v_B, %v_C, %v_D)
          : (vector<8xi64>, vector<8xf64>, vector<8xi64>, vector<8xf64>) -> f64
      %r2 = addf %sum1, %subresult : f64
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
func @memref_dot_optimized(%m_A : memref<?xi64>, %m_B : memref<?xf64>,
                           %m_C : memref<?xi64>, %m_D : memref<?xf64>,
                           %M : index, %N : index)
    -> f64 {
  // Helper constants for loops.
  %c0 = constant 0 : index
  %i0 = constant 0 : i32
  %i7 = constant 7 : i32
  %c8 = constant 8 : index

  %data_zero = constant 0.0 : f64
  %index_padding = constant 9223372036854775807 : i64

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
      %seg1_done = cmpi "slt", %segB_max, %segA_min : i64

      %r2, %next_b_start1 = scf.if %seg1_done -> (f64, index) {
        // %v_C segment is done, no need to examine this one again (ever).
        %next_b_start2 = addi %b_start1, %c8 : index
        scf.yield %sum1, %next_b_start2 : f64, index
      } else {
        %v_B = vector.transfer_read %m_B[%a], %data_zero
            : memref<?xf64>, vector<8xf64>
        %v_D = vector.transfer_read %m_D[%b], %data_zero
            : memref<?xf64>, vector<8xf64>

        %subresult = call @vector_dot(%v_A, %v_B, %v_C, %v_D)
            : (vector<8xi64>, vector<8xf64>, vector<8xi64>, vector<8xf64>)
                -> f64
        %r3 = addf %sum1, %subresult : f64
        scf.yield %r3, %b_start1 : f64, index
      }

      scf.yield %r2, %next_b_start1 : f64, index
    }

    scf.yield %r1, %next_b_start0 : f64, index
  }

  return %r0 : f64
}

func @entry() -> i32 {
  // Initialize large buffers that can be used for multiple test cases of
  // different sizes.
  %b_A = alloc() : memref<128xi64>
  %b_B = alloc() : memref<128xf64>
  %b_C = alloc() : memref<128xi64>
  %b_D = alloc() : memref<128xf64>

  %m_A = memref_cast %b_A : memref<128xi64> to memref<?xi64>
  %m_B = memref_cast %b_B : memref<128xf64> to memref<?xf64>
  %m_C = memref_cast %b_C : memref<128xi64> to memref<?xi64>
  %m_D = memref_cast %b_D : memref<128xf64> to memref<?xf64>

  // --- Test case 1 ---.
  // M and N must be a multiple of 8 if smaller than 128.
  // (Because padding kicks in only for out-of-bounds accesses.)
  %M1, %N1 = call @fill_input_1(%m_A, %m_B, %m_C, %m_D)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>)
          -> (index, index)

  %r0 = call @memref_dot_simple(%m_A, %m_B, %m_C, %m_D, %M1, %N1)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r0 : f64
  // CHECK: 86

  %r1 = call @memref_dot_optimized(%m_A, %m_B, %m_C, %m_D, %M1, %N1)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r1 : f64
  // CHECK: 86

  // --- Test case 2 ---.
  // M and N must be a multiple of 8 if smaller than 128.
  // (Because padding kicks in only for out-of-bounds accesses.)
  %M2, %N2 = call @fill_input_2(%m_A, %m_B, %m_C, %m_D)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>)
          -> (index, index)

  %r3 = call @memref_dot_simple(%m_A, %m_B, %m_C, %m_D, %M2, %N2)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r3 : f64
  // CHECK: 111

  %r4 = call @memref_dot_optimized(%m_A, %m_B, %m_C, %m_D, %M2, %N2)
      : (memref<?xi64>, memref<?xf64>, memref<?xi64>, memref<?xf64>,
         index, index) -> f64
  vector.print %r4 : f64
  // CHECK: 111

  // Release all resources.
  dealloc %b_A : memref<128xi64>
  dealloc %b_B : memref<128xf64>
  dealloc %b_C : memref<128xi64>
  dealloc %b_D : memref<128xf64>

  %r = constant 0 : i32
  return %r : i32
}

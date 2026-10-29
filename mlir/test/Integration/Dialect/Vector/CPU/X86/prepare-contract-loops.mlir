// RUN: mlir-opt %s -test-transform-dialect-erase-schedule \
// RUN:   -convert-vector-to-llvm='vector-contract-lowering=outerproduct' \
// RUN:   -test-lower-to-llvm | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils | FileCheck %s
// RUN: mlir-opt %s -transform-interpreter \
// RUN:   -test-transform-dialect-erase-schedule \
// RUN:   -convert-vector-to-llvm='vector-contract-lowering=outerproduct' \
// RUN:   -test-lower-to-llvm | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils | FileCheck %s

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>

// Generic CPU lowering executes VNNI contractions only without mixed precision.
func.func @vnni_bf16(%a: memref<1x2x8xbf16>,
                     %b: memref<1x4x3x2xbf16>) -> vector<2x3xbf16> {
  %zero = arith.constant 0 : index
  %end = arith.constant 1 : index
  %pad = arith.constant 0.0 : bf16
  %init = arith.constant dense<0.0> : vector<2x3xbf16>
  %result = scf.for %k = %zero to %end step %end iter_args(%accum = %init) -> (vector<2x3xbf16>) {
    %left = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x2x8xbf16>, vector<2x8xbf16>
    %right = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<1x4x3x2xbf16>, vector<4x3x2xbf16>
    %transpose = vector.transpose %right, [0, 2, 1] : vector<4x3x2xbf16> to vector<4x2x3xbf16>
    %flat = vector.shape_cast %transpose : vector<4x2x3xbf16> to vector<8x3xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %left, %flat, %accum : vector<2x8xbf16>, vector<8x3xbf16> into vector<2x3xbf16>
    scf.yield %dot : vector<2x3xbf16>
  }
  return %result : vector<2x3xbf16>
}

func.func @vnni_i8(%a: memref<1x2x8xi8>,
                   %b: memref<1x2x3x4xi8>) -> vector<2x3xi8> {
  %zero = arith.constant 0 : index
  %end = arith.constant 1 : index
  %pad = arith.constant 0 : i8
  %init = arith.constant dense<0> : vector<2x3xi8>
  %result = scf.for %k = %zero to %end step %end iter_args(%accum = %init) -> (vector<2x3xi8>) {
    %left = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x2x8xi8>, vector<2x8xi8>
    %right = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<1x2x3x4xi8>, vector<2x3x4xi8>
    %transpose = vector.transpose %right, [0, 2, 1] : vector<2x3x4xi8> to vector<2x4x3xi8>
    %flat = vector.shape_cast %transpose : vector<2x4x3xi8> to vector<8x3xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %left, %flat, %accum : vector<2x8xi8>, vector<8x3xi8> into vector<2x3xi8>
    scf.yield %dot : vector<2x3xi8>
  }
  return %result : vector<2x3xi8>
}

// Check that folding deinterleave/transpose/interleave/transpose unpacking
// preserves the matmul result.
func.func @interleaved_bf16(%a: memref<1x2x8xbf16>,
                            %b: memref<1x4x6xbf16>) -> vector<2x3xbf16> {
  %zero = arith.constant 0 : index
  %end = arith.constant 1 : index
  %pad = arith.constant 0.0 : bf16
  %init = arith.constant dense<0.0> : vector<2x3xbf16>
  %result = scf.for %k = %zero to %end step %end iter_args(%accum = %init) -> (vector<2x3xbf16>) {
    %left = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x2x8xbf16>, vector<2x8xbf16>
    %right = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x4x6xbf16>, vector<4x6xbf16>
    %even, %odd = vector.deinterleave %right : vector<4x6xbf16> -> vector<4x3xbf16>
    %even_t = vector.transpose %even, [1, 0] : vector<4x3xbf16> to vector<3x4xbf16>
    %odd_t = vector.transpose %odd, [1, 0] : vector<4x3xbf16> to vector<3x4xbf16>
    %interleaved = vector.interleave %even_t, %odd_t : vector<3x4xbf16> -> vector<3x8xbf16>
    %flat = vector.transpose %interleaved, [1, 0] : vector<3x8xbf16> to vector<8x3xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %left, %flat, %accum : vector<2x8xbf16>, vector<8x3xbf16> into vector<2x3xbf16>
    scf.yield %dot : vector<2x3xbf16>
  }
  return %result : vector<2x3xbf16>
}

// Check the equivalent transpose/extract and slice-insert form.
func.func @split_join_bf16(%a: memref<1x2x8xbf16>,
                           %b: memref<1x4x6xbf16>) -> vector<2x3xbf16> {
  %zero = arith.constant 0 : index
  %end = arith.constant 1 : index
  %pad = arith.constant 0.0 : bf16
  %poison = ub.poison : vector<3x4x2xbf16>
  %init = arith.constant dense<0.0> : vector<2x3xbf16>
  %result = scf.for %k = %zero to %end step %end iter_args(%accum = %init) -> (vector<2x3xbf16>) {
    %left = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x2x8xbf16>, vector<2x8xbf16>
    %right = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x4x6xbf16>, vector<4x6xbf16>
    %packed = vector.shape_cast %right : vector<4x6xbf16> to vector<4x3x2xbf16>
    %even_odd = vector.transpose %packed, [2, 0, 1] : vector<4x3x2xbf16> to vector<2x4x3xbf16>
    %even = vector.extract %even_odd[0] : vector<4x3xbf16> from vector<2x4x3xbf16>
    %odd = vector.extract %even_odd[1] : vector<4x3xbf16> from vector<2x4x3xbf16>
    %even_t = vector.transpose %even, [1, 0] : vector<4x3xbf16> to vector<3x4xbf16>
    %odd_t = vector.transpose %odd, [1, 0] : vector<4x3xbf16> to vector<3x4xbf16>
    %even3 = vector.shape_cast %even_t : vector<3x4xbf16> to vector<3x4x1xbf16>
    %odd3 = vector.shape_cast %odd_t : vector<3x4xbf16> to vector<3x4x1xbf16>
    %join0 = vector.insert_strided_slice %even3, %poison offsets = [0, 0, 0], strides = [1, 1, 1] : vector<3x4x1xbf16> into vector<3x4x2xbf16>
    %join = vector.insert_strided_slice %odd3, %join0 offsets = [0, 0, 1], strides = [1, 1, 1] : vector<3x4x1xbf16> into vector<3x4x2xbf16>
    %interleaved = vector.shape_cast %join : vector<3x4x2xbf16> to vector<3x8xbf16>
    %flat = vector.transpose %interleaved, [1, 0] : vector<3x8xbf16> to vector<8x3xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %left, %flat, %accum : vector<2x8xbf16>, vector<8x3xbf16> into vector<2x3xbf16>
    scf.yield %dot : vector<2x3xbf16>
  }
  return %result : vector<2x3xbf16>
}

// Check i8 unpacking through two levels of binary splits and joins. The first
// split yields VNNI lanes {0, 2} and {1, 3}; mix deinterleave with
// transpose/extract, and interleave with out-of-order slice inserts.
func.func @split_join_i8(%a: memref<1x2x8xi8>,
                         %b: memref<1x2x12xi8>) -> vector<2x3xi8> {
  %zero = arith.constant 0 : index
  %end = arith.constant 1 : index
  %pad = arith.constant 0 : i8
  %poison = ub.poison : vector<3x2x2xi8>
  %init = arith.constant dense<0> : vector<2x3xi8>
  %result = scf.for %k = %zero to %end step %end iter_args(%accum = %init) -> (vector<2x3xi8>) {
    %left = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x2x8xi8>, vector<2x8xi8>
    %right = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x2x12xi8>, vector<2x12xi8>
    %even, %odd = vector.deinterleave %right : vector<2x12xi8> -> vector<2x6xi8>
    %lane0, %lane2 = vector.deinterleave %even : vector<2x6xi8> -> vector<2x3xi8>
    %odd_packed = vector.shape_cast %odd : vector<2x6xi8> to vector<2x3x2xi8>
    %odd_split = vector.transpose %odd_packed, [2, 0, 1] : vector<2x3x2xi8> to vector<2x2x3xi8>
    %lane1 = vector.extract %odd_split[0] : vector<2x3xi8> from vector<2x2x3xi8>
    %lane3 = vector.extract %odd_split[1] : vector<2x3xi8> from vector<2x2x3xi8>
    %t0 = vector.transpose %lane0, [1, 0] : vector<2x3xi8> to vector<3x2xi8>
    %t1 = vector.transpose %lane1, [1, 0] : vector<2x3xi8> to vector<3x2xi8>
    %t2 = vector.transpose %lane2, [1, 0] : vector<2x3xi8> to vector<3x2xi8>
    %t3 = vector.transpose %lane3, [1, 0] : vector<2x3xi8> to vector<3x2xi8>
    %even_t = vector.interleave %t0, %t2 : vector<3x2xi8> -> vector<3x4xi8>
    %s1 = vector.shape_cast %t1 : vector<3x2xi8> to vector<3x2x1xi8>
    %s3 = vector.shape_cast %t3 : vector<3x2xi8> to vector<3x2x1xi8>
    %join0 = vector.insert_strided_slice %s3, %poison offsets = [0, 0, 1], strides = [1, 1, 1] : vector<3x2x1xi8> into vector<3x2x2xi8>
    %join1 = vector.insert_strided_slice %s1, %join0 offsets = [0, 0, 0], strides = [1, 1, 1] : vector<3x2x1xi8> into vector<3x2x2xi8>
    %odd_t = vector.shape_cast %join1 : vector<3x2x2xi8> to vector<3x4xi8>
    %joined = vector.interleave %even_t, %odd_t : vector<3x4xi8> -> vector<3x8xi8>
    %flat = vector.transpose %joined, [1, 0] : vector<3x8xi8> to vector<8x3xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %left, %flat, %accum : vector<2x8xi8>, vector<8x3xi8> into vector<2x3xi8>
    scf.yield %dot : vector<2x3xi8>
  }
  return %result : vector<2x3xi8>
}

// Check i8 unpacking through a single 4-way transpose/extract split and a
// 4-way slice-insert join with out-of-order inserts.
func.func @fourway_split_join_i8(%a: memref<1x2x8xi8>,
                                 %b: memref<1x2x12xi8>) -> vector<2x3xi8> {
  %zero = arith.constant 0 : index
  %end = arith.constant 1 : index
  %pad = arith.constant 0 : i8
  %poison = ub.poison : vector<3x2x4xi8>
  %init = arith.constant dense<0> : vector<2x3xi8>
  %result = scf.for %k = %zero to %end step %end iter_args(%accum = %init) -> (vector<2x3xi8>) {
    %left = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x2x8xi8>, vector<2x8xi8>
    %right = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x2x12xi8>, vector<2x12xi8>
    %packed = vector.shape_cast %right : vector<2x12xi8> to vector<2x3x4xi8>
    %split = vector.transpose %packed, [2, 0, 1] : vector<2x3x4xi8> to vector<4x2x3xi8>
    %lane0 = vector.extract %split[0] : vector<2x3xi8> from vector<4x2x3xi8>
    %lane1 = vector.extract %split[1] : vector<2x3xi8> from vector<4x2x3xi8>
    %lane2 = vector.extract %split[2] : vector<2x3xi8> from vector<4x2x3xi8>
    %lane3 = vector.extract %split[3] : vector<2x3xi8> from vector<4x2x3xi8>
    %t0 = vector.transpose %lane0, [1, 0] : vector<2x3xi8> to vector<3x2xi8>
    %t1 = vector.transpose %lane1, [1, 0] : vector<2x3xi8> to vector<3x2xi8>
    %t2 = vector.transpose %lane2, [1, 0] : vector<2x3xi8> to vector<3x2xi8>
    %t3 = vector.transpose %lane3, [1, 0] : vector<2x3xi8> to vector<3x2xi8>
    %s0 = vector.shape_cast %t0 : vector<3x2xi8> to vector<3x2x1xi8>
    %s1 = vector.shape_cast %t1 : vector<3x2xi8> to vector<3x2x1xi8>
    %s2 = vector.shape_cast %t2 : vector<3x2xi8> to vector<3x2x1xi8>
    %s3 = vector.shape_cast %t3 : vector<3x2xi8> to vector<3x2x1xi8>
    %join0 = vector.insert_strided_slice %s2, %poison offsets = [0, 0, 2], strides = [1, 1, 1] : vector<3x2x1xi8> into vector<3x2x4xi8>
    %join1 = vector.insert_strided_slice %s0, %join0 offsets = [0, 0, 0], strides = [1, 1, 1] : vector<3x2x1xi8> into vector<3x2x4xi8>
    %join2 = vector.insert_strided_slice %s3, %join1 offsets = [0, 0, 3], strides = [1, 1, 1] : vector<3x2x1xi8> into vector<3x2x4xi8>
    %join3 = vector.insert_strided_slice %s1, %join2 offsets = [0, 0, 1], strides = [1, 1, 1] : vector<3x2x1xi8> into vector<3x2x4xi8>
    %joined = vector.shape_cast %join3 : vector<3x2x4xi8> to vector<3x8xi8>
    %flat = vector.transpose %joined, [1, 0] : vector<3x8xi8> to vector<8x3xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %left, %flat, %accum : vector<2x8xi8>, vector<8x3xi8> into vector<2x3xi8>
    scf.yield %dot : vector<2x3xi8>
  }
  return %result : vector<2x3xi8>
}

func.func @main() {
  %zero = arith.constant 0 : index

  // All functions unpack the same logical 8x3 RHS from either 2-way (bf16) or
  // 4-way (i8) VNNI layout and multiply it by the same 2x8 LHS. Unequal VNNI
  // lanes and rectangular shapes expose lane permutations and wrong reshapes.
  %af = memref.alloc() : memref<1x2x8xbf16>
  %bf = memref.alloc() : memref<1x4x6xbf16>
  %bf_packed = memref.expand_shape %bf [[0], [1], [2, 3]] output_shape [1, 4, 3, 2] : memref<1x4x6xbf16> into memref<1x4x3x2xbf16>
  %lhs_bf = arith.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                  [2.0, 0.0, 1.0, 3.0, 0.0, 2.0, 1.0, 3.0]]> : vector<2x8xbf16>
  %rhs_bf = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 1.0, 0.0],
                                  [2.0, 0.0, 3.0, 1.0, 0.0, 2.0],
                                  [1.0, 3.0, 0.0, 0.0, 2.0, 1.0],
                                  [3.0, 2.0, 1.0, 0.0, 1.0, 3.0]]> : vector<4x6xbf16>
  vector.transfer_write %lhs_bf, %af[%zero, %zero, %zero] {in_bounds = [true, true]} : vector<2x8xbf16>, memref<1x2x8xbf16>
  vector.transfer_write %rhs_bf, %bf[%zero, %zero, %zero] {in_bounds = [true, true]} : vector<4x6xbf16>, memref<1x4x6xbf16>
  %vnni_bf16 = func.call @vnni_bf16(%af, %bf_packed) : (memref<1x2x8xbf16>, memref<1x4x3x2xbf16>) -> vector<2x3xbf16>
  vector.print %vnni_bf16 : vector<2x3xbf16>
  // CHECK: ( ( 68, 28, 56 ), ( 17, 11, 20 ) )
  %interleaved = func.call @interleaved_bf16(%af, %bf) : (memref<1x2x8xbf16>, memref<1x4x6xbf16>) -> vector<2x3xbf16>
  vector.print %interleaved : vector<2x3xbf16>
  // CHECK-NEXT: ( ( 68, 28, 56 ), ( 17, 11, 20 ) )
  %split_join = func.call @split_join_bf16(%af, %bf) : (memref<1x2x8xbf16>, memref<1x4x6xbf16>) -> vector<2x3xbf16>
  vector.print %split_join : vector<2x3xbf16>
  // CHECK-NEXT: ( ( 68, 28, 56 ), ( 17, 11, 20 ) )

  %ai = memref.alloc() : memref<1x2x8xi8>
  %bi = memref.alloc() : memref<1x2x12xi8>
  %bi_packed = memref.expand_shape %bi [[0], [1], [2, 3]] output_shape [1, 2, 3, 4] : memref<1x2x12xi8> into memref<1x2x3x4xi8>
  %lhs_i = arith.constant dense<[[1, 2, 3, 4, 5, 6, 7, 8],
                                 [2, 0, 1, 3, 0, 2, 1, 3]]> : vector<2x8xi8>
  %rhs_i = arith.constant dense<[[0, 1, 2, 0, 2, 3, 3, 1, 1, 0, 0, 2],
                                 [1, 3, 3, 2, 0, 0, 1, 0, 2, 1, 1, 3]]> : vector<2x12xi8>
  vector.transfer_write %lhs_i, %ai[%zero, %zero, %zero] {in_bounds = [true, true]} : vector<2x8xi8>, memref<1x2x8xi8>
  vector.transfer_write %rhs_i, %bi[%zero, %zero, %zero] {in_bounds = [true, true]} : vector<2x12xi8>, memref<1x2x12xi8>
  %vnni_i8 = func.call @vnni_i8(%ai, %bi_packed) : (memref<1x2x8xi8>, memref<1x2x3x4xi8>) -> vector<2x3xi8>
  vector.print %vnni_i8 : vector<2x3xi8>
  // CHECK-NEXT: ( ( 68, 28, 56 ), ( 17, 11, 20 ) )
  %split_join_i8 = func.call @split_join_i8(%ai, %bi) : (memref<1x2x8xi8>, memref<1x2x12xi8>) -> vector<2x3xi8>
  vector.print %split_join_i8 : vector<2x3xi8>
  // CHECK-NEXT: ( ( 68, 28, 56 ), ( 17, 11, 20 ) )
  %fourway_i8 = func.call @fourway_split_join_i8(%ai, %bi) : (memref<1x2x8xi8>, memref<1x2x12xi8>) -> vector<2x3xi8>
  vector.print %fourway_i8 : vector<2x3xi8>
  // CHECK-NEXT: ( ( 68, 28, 56 ), ( 17, 11, 20 ) )
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.x86.prepare_vector_contract_loops
    } : !transform.any_op
    transform.yield
  }
}

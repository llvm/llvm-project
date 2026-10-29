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
                     %b: memref<1x4x4x2xbf16>) -> vector<2x4xbf16> {
  %zero = arith.constant 0 : index
  %end = arith.constant 1 : index
  %pad = arith.constant 0.0 : bf16
  %init = arith.constant dense<0.0> : vector<2x4xbf16>
  %result = scf.for %k = %zero to %end step %end iter_args(%accum = %init) -> (vector<2x4xbf16>) {
    %left = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x2x8xbf16>, vector<2x8xbf16>
    %right = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<1x4x4x2xbf16>, vector<4x4x2xbf16>
    %transpose = vector.transpose %right, [0, 2, 1] : vector<4x4x2xbf16> to vector<4x2x4xbf16>
    %flat = vector.shape_cast %transpose : vector<4x2x4xbf16> to vector<8x4xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %left, %flat, %accum : vector<2x8xbf16>, vector<8x4xbf16> into vector<2x4xbf16>
    scf.yield %dot : vector<2x4xbf16>
  }
  return %result : vector<2x4xbf16>
}

func.func @vnni_i8(%a: memref<1x2x8xi8>,
                   %b: memref<1x2x4x4xi8>) -> vector<2x4xi8> {
  %zero = arith.constant 0 : index
  %end = arith.constant 1 : index
  %pad = arith.constant 0 : i8
  %init = arith.constant dense<0> : vector<2x4xi8>
  %result = scf.for %k = %zero to %end step %end iter_args(%accum = %init) -> (vector<2x4xi8>) {
    %left = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<1x2x8xi8>, vector<2x8xi8>
    %right = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<1x2x4x4xi8>, vector<2x4x4xi8>
    %transpose = vector.transpose %right, [0, 2, 1] : vector<2x4x4xi8> to vector<2x4x4xi8>
    %flat = vector.shape_cast %transpose : vector<2x4x4xi8> to vector<8x4xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %left, %flat, %accum : vector<2x8xi8>, vector<8x4xi8> into vector<2x4xi8>
    scf.yield %dot : vector<2x4xi8>
  }
  return %result : vector<2x4xi8>
}

func.func @main() {
  %zero = arith.constant 0 : index
  %af = memref.alloc() : memref<1x2x8xbf16>
  %bf = memref.alloc() : memref<1x4x4x2xbf16>
  %ones_af = arith.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]> : vector<2x8xbf16>
  %ones_bf = arith.constant dense<[
    [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]],
    [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]],
    [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]],
    [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]
  ]> : vector<4x4x2xbf16>
  vector.transfer_write %ones_af, %af[%zero, %zero, %zero] {in_bounds = [true, true]} : vector<2x8xbf16>, memref<1x2x8xbf16>
  vector.transfer_write %ones_bf, %bf[%zero, %zero, %zero, %zero] {in_bounds = [true, true, true]} : vector<4x4x2xbf16>, memref<1x4x4x2xbf16>
  %bf_result = func.call @vnni_bf16(%af, %bf) : (memref<1x2x8xbf16>, memref<1x4x4x2xbf16>) -> vector<2x4xbf16>
  %bf_value = vector.extract %bf_result[0, 0] : bf16 from vector<2x4xbf16>
  vector.print %bf_value : bf16
  // CHECK: 92

  %a4 = memref.alloc() : memref<1x2x8xi8>
  %b4 = memref.alloc() : memref<1x2x4x4xi8>
  %ones_a4 = arith.constant dense<[[1, 2, 3, 4, 5, 6, 7, 8],
                                   [1, 2, 3, 4, 5, 6, 7, 8]]> : vector<2x8xi8>
  %ones_b4 = arith.constant dense<[
    [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
  ]> : vector<2x4x4xi8>
  vector.transfer_write %ones_a4, %a4[%zero, %zero, %zero] {in_bounds = [true, true]} : vector<2x8xi8>, memref<1x2x8xi8>
  vector.transfer_write %ones_b4, %b4[%zero, %zero, %zero, %zero] {in_bounds = [true, true, true]} : vector<2x4x4xi8>, memref<1x2x4x4xi8>
  %i8_result = func.call @vnni_i8(%a4, %b4) : (memref<1x2x8xi8>, memref<1x2x4x4xi8>) -> vector<2x4xi8>
  %i8_value = vector.extract %i8_result[0, 0] : i8 from vector<2x4xi8>
  vector.print %i8_value : i8
  // CHECK-NEXT: 100
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

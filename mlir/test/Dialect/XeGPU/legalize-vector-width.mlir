// RUN: mlir-opt %s --xegpu-legalize-vector-width --split-input-file | FileCheck %s
// RUN: mlir-opt %s --xegpu-legalize-vector-width=max-vector-components=8 --split-input-file | FileCheck %s --check-prefix=CHECK8

// Elementwise ops wider than the target width are split; the wide value only
// survives at the non-elementwise boundary.

// CHECK-LABEL: func.func @split_elementwise
// CHECK-COUNT-2: arith.divf {{.*}} : vector<16xbf16>
// CHECK-NOT: arith.divf {{.*}} : vector<32xbf16>

// CHECK8-LABEL: func.func @split_elementwise
// CHECK8-COUNT-4: arith.divf {{.*}} : vector<8xbf16>
// CHECK8-NOT: arith.divf {{.*}} : vector<32xbf16>
func.func @split_elementwise(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  %0 = arith.divf %a, %b : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// Already-narrow ops are left untouched.

// CHECK-LABEL: func.func @already_legal
// CHECK-NEXT: arith.divf {{.*}} : vector<16xbf16>
// CHECK-NEXT: return
func.func @already_legal(%a: vector<16xbf16>, %b: vector<16xbf16>) -> vector<16xbf16> {
  %0 = arith.divf %a, %b : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// -----

// Ops changing element width are still elementwise and must be split, tracking
// the result shape.

// CHECK-LABEL: func.func @split_cast
// CHECK-COUNT-2: arith.extui {{.*}} : vector<16xi8> to vector<16xi32>
// CHECK-NOT: arith.extui {{.*}} : vector<32xi8> to vector<32xi32>
func.func @split_cast(%a: vector<32xi8>) -> vector<32xi32> {
  %0 = arith.extui %a : vector<32xi8> to vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// `vector.shuffle` is data movement, not elementwise: a wide shuffle is a
// packed payload and must be preserved.

// CHECK-LABEL: func.func @shuffle_untouched
// CHECK-NEXT: vector.shuffle
// CHECK-NEXT: return
func.func @shuffle_untouched(%a: vector<32xf4E2M1FN>, %b: vector<32xf4E2M1FN>) -> vector<64xf4E2M1FN> {
  %0 = vector.shuffle %a, %b [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                              8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47,
                              16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55,
                              24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63]
     : vector<32xf4E2M1FN>, vector<32xf4E2M1FN>
  return %0 : vector<64xf4E2M1FN>
}

// -----

// A length-changing `vector.bitcast` is not elementwise either.

// CHECK-LABEL: func.func @bitcast_untouched
// CHECK-NEXT: vector.bitcast {{.*}} : vector<32xi8> to vector<64xf4E2M1FN>
// CHECK-NEXT: return
func.func @bitcast_untouched(%a: vector<32xi8>) -> vector<64xf4E2M1FN> {
  %0 = vector.bitcast %a : vector<32xi8> to vector<64xf4E2M1FN>
  return %0 : vector<64xf4E2M1FN>
}

// -----

// Sub-byte results are packed payloads, not compute values. Splitting the
// quantizing `arith.truncf` would glue the halves back together with
// `insert_strided_slice` on `f4E2M1FN`, which lowers to untranslatable
// sub-byte vector movement. Its own XeVM lowering handles width instead.

// CHECK-LABEL: func.func @subbyte_result_untouched
// CHECK-NEXT: arith.truncf {{.*}} : vector<32xbf16> to vector<32xf4E2M1FN>
// CHECK-NEXT: return
func.func @subbyte_result_untouched(%a: vector<32xbf16>) -> vector<32xf4E2M1FN> {
  %0 = arith.truncf %a : vector<32xbf16> to vector<32xf4E2M1FN>
  return %0 : vector<32xf4E2M1FN>
}

// -----

// Same in the dequantizing direction, where the sub-byte type is the operand.

// CHECK-LABEL: func.func @subbyte_operand_untouched
// CHECK-NEXT: arith.extf {{.*}} : vector<32xf4E2M1FN> to vector<32xbf16>
// CHECK-NEXT: return
func.func @subbyte_operand_untouched(%a: vector<32xf4E2M1FN>) -> vector<32xbf16> {
  %0 = arith.extf %a : vector<32xf4E2M1FN> to vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// `i1` masks are ordinary compute values and must still be split.

// CHECK-LABEL: func.func @mask_is_not_subbyte
// CHECK-COUNT-2: arith.cmpi eq, {{.*}} : vector<16xi8>
// CHECK-NOT: arith.cmpi eq, {{.*}} : vector<32xi8>
func.func @mask_is_not_subbyte(%a: vector<32xi8>, %b: vector<32xi8>) -> vector<32xi1> {
  %0 = arith.cmpi eq, %a, %b : vector<32xi8>
  return %0 : vector<32xi1>
}

// -----

// Widths that are not a multiple of the limit fall back to the largest divisor
// so that no ragged tail is produced. For 24 components with a limit of 16 the
// largest usable divisor is 12.

// CHECK-LABEL: func.func @non_multiple_width
// CHECK-COUNT-2: arith.negf {{.*}} : vector<12xf32>
// CHECK-NOT: arith.negf {{.*}} : vector<24xf32>
func.func @non_multiple_width(%a: vector<24xf32>) -> vector<24xf32> {
  %0 = arith.negf %a : vector<24xf32>
  return %0 : vector<24xf32>
}

// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

/// Note: Inspect generated assembly and llvm-mca stats:
/// ====================================================
/// mlir-opt --transform-interpreter mlir/test/Dialect/Vector/vector-rewrite-narrow-types.mlir -test-transform-dialect-erase-schedule -test-lower-to-llvm | mlir-translate -mlir-to-llvmir | llc -o - -mcpu=skylake-avx512 --function-sections -filetype=obj > /tmp/a.out; objdump -d --disassemble=f1 --no-addresses --no-show-raw-insn -M att /tmp/a.out | ./build/bin/llvm-mca -mcpu=skylake-avx512

// CHECK-LABEL: func.func @f1(
//  CHECK-SAME: %[[A:[0-9a-z]*]]: vector<32xi64>) -> vector<20xi8>
func.func @f1(%a: vector<32xi64>) -> vector<20xi8> {
  /// Rewriting this standalone pattern is about 2x faster on skylake-ax512 according to llvm-mca.
  /// Benefit further increases when mixed with other compute ops.
  ///
  /// The provenance of the 20x8 bits of the result are the following bits in the
  /// source vector:
  // { 0: b@[0..5) lshl: 0 } { 1: b@[0..3) lshl: 5 }
  // { 1: b@[3..5) lshl: 0 } { 2: b@[0..5) lshl: 2 } { 3: b@[0..1) lshl: 7 }
  // { 3: b@[1..5) lshl: 0 } { 4: b@[0..4) lshl: 4 }
  // { 4: b@[4..5) lshl: 0 } { 5: b@[0..5) lshl: 1 } { 6: b@[0..2) lshl: 6 }
  // { 6: b@[2..5) lshl: 0 } { 7: b@[0..5) lshl: 3 }
  // { 8: b@[0..5) lshl: 0 } { 9: b@[0..3) lshl: 5 }
  // { 9: b@[3..5) lshl: 0 } { 10: b@[0..5) lshl: 2 } { 11: b@[0..1) lshl: 7 }
  // { 11: b@[1..5) lshl: 0 } { 12: b@[0..4) lshl: 4 }
  // { 12: b@[4..5) lshl: 0 } { 13: b@[0..5) lshl: 1 } { 14: b@[0..2) lshl: 6 }
  // { 14: b@[2..5) lshl: 0 } { 15: b@[0..5) lshl: 3 }
  // { 16: b@[0..5) lshl: 0 } { 17: b@[0..3) lshl: 5 }
  // { 17: b@[3..5) lshl: 0 } { 18: b@[0..5) lshl: 2 } { 19: b@[0..1) lshl: 7 }
  // { 19: b@[1..5) lshl: 0 } { 20: b@[0..4) lshl: 4 }
  // { 20: b@[4..5) lshl: 0 } { 21: b@[0..5) lshl: 1 } { 22: b@[0..2) lshl: 6 }
  // { 22: b@[2..5) lshl: 0 } { 23: b@[0..5) lshl: 3 }
  // { 24: b@[0..5) lshl: 0 } { 25: b@[0..3) lshl: 5 }
  // { 25: b@[3..5) lshl: 0 } { 26: b@[0..5) lshl: 2 } { 27: b@[0..1) lshl: 7 }
  // { 27: b@[1..5) lshl: 0 } { 28: b@[0..4) lshl: 4 }
  // { 28: b@[4..5) lshl: 0 } { 29: b@[0..5) lshl: 1 } { 30: b@[0..2) lshl: 6 }
  // { 30: b@[2..5) lshl: 0 } { 31: b@[0..5) lshl: 3 }
  /// This results in 3 shuffles + 1 shr + 2 shl + 3 and + 2 or.
  /// The third vector is empty for positions 0, 2, 4, 5, 7, 9, 10, 12, 14, 15,
  /// 17 and 19 (i.e. there are only 2 entries in that row).
  ///
  ///                             0: b@[0..5), 1: b@[3..5), etc
  // CHECK-DAG: %[[MASK0:.*]] = arith.constant dense<[31, 24, 30, 16, 28, 31, 24, 30, 16, 28, 31, 24, 30, 16, 28, 31, 24, 30, 16, 28]> : vector<20xi64>
  ///                             1: b@[0..3), 2: b@[0..5), etc
  // CHECK-DAG: %[[MASK1:.*]] = arith.constant dense<[7, 31, 15, 31, 31, 7, 31, 15, 31, 31, 7, 31, 15, 31, 31, 7, 31, 15, 31, 31]> :  vector<20xi64>
  ///                             empty, 3: b@[0..1), empty etc
  // CHECK-DAG: %[[MASK2:.*]] = arith.constant dense<[0, 1, 0, 3, 0, 0, 1, 0, 3, 0, 0, 1, 0, 3, 0, 0, 1, 0, 3, 0]> : vector<20xi64>
  // CHECK-DAG: %[[SHR0_CST:.*]] = arith.constant dense<[0, 3, 1, 4, 2, 0, 3, 1, 4, 2, 0, 3, 1, 4, 2, 0, 3, 1, 4, 2]> : vector<20xi64>
  // CHECK-DAG: %[[SHL1_CST:.*]] = arith.constant dense<[5, 2, 4, 1, 3, 5, 2, 4, 1, 3, 5, 2, 4, 1, 3, 5, 2, 4, 1, 3]> : vector<20xi64>
  // CHECK-DAG: %[[SHL2_CST:.*]] = arith.constant dense<[8, 7, 8, 6, 8, 8, 7, 8, 6, 8, 8, 7, 8, 6, 8, 8, 7, 8, 6, 8]> : vector<20xi64>
  //
  // CHECK: %[[V0:.*]] = vector.shuffle %[[A]], %[[A]] [0, 1, 3, 4, 6, 8, 9, 11, 12, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 30] : vector<32xi64>, vector<32xi64>
  // CHECK: %[[A0:.*]] = arith.andi %[[V0]], %[[MASK0]] : vector<20xi64>
  // CHECK: %[[SHR0:.*]] = arith.shrui %[[A0]], %[[SHR0_CST]] : vector<20xi64>
  // CHECK: %[[V1:.*]] = vector.shuffle %[[A]], %[[A]] [1, 2, 4, 5, 7, 9, 10, 12, 13, 15, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31] : vector<32xi64>, vector<32xi64>
  // CHECK: %[[A1:.*]] = arith.andi %[[V1]], %[[MASK1]] : vector<20xi64>
  // CHECK: %[[SHL1:.*]] = arith.shli %[[A1]], %[[SHL1_CST]] : vector<20xi64>
  // CHECK: %[[O1:.*]] = arith.ori %[[SHR0]], %[[SHL1]] : vector<20xi64>
  // CHECK: %[[V2:.*]] = vector.shuffle %[[A]], %[[A]] [0, 3, 0, 6, 0, 0, 11, 0, 14, 0, 0, 19, 0, 22, 0, 0, 27, 0, 30, 0] : vector<32xi64>, vector<32xi64>
  // CHECK: %[[A2:.*]] = arith.andi %[[V2]], %[[MASK2]] : vector<20xi64>
  // CHECK: %[[SHL2:.*]] = arith.shli %[[A2]], %[[SHL2_CST]] : vector<20xi64>
  // CHECK: %[[O2:.*]] = arith.ori %[[O1]], %[[SHL2]] : vector<20xi64>
  // CHECK: %[[TR:.*]] = arith.trunci %[[O2]] : vector<20xi64> to vector<20xi8>
  // CHECK-NOT: bitcast
  %0 = arith.trunci %a : vector<32xi64> to vector<32xi5>
  %1 = vector.bitcast %0 : vector<32xi5> to vector<20xi8>
  return %1 : vector<20xi8>
}

// CHECK-LABEL: func.func @f2(
//  CHECK-SAME:   %[[A:[0-9a-z]*]]: vector<16xi16>) -> vector<3xi16>
func.func @f2(%a: vector<16xi16>) -> vector<3xi16> {
  /// Rewriting this standalone pattern is about 1.8x faster on skylake-ax512 according to llvm-mca.
  /// Benefit further increases when mixed with other compute ops.
  ///
  // { 0: b@[0..3) lshl: 0 } { 1: b@[0..3) lshl: 3 } { 2: b@[0..3) lshl: 6 } { 3: b@[0..3) lshl: 9 } { 4: b@[0..3) lshl: 12 } { 5: b@[0..1) lshl: 15 }
  // { 5: b@[1..3) lshl: 0 } { 6: b@[0..3) lshl: 2 } { 7: b@[0..3) lshl: 5 } { 8: b@[0..3) lshl: 8 } { 9: b@[0..3) lshl: 11 } { 10: b@[0..2) lshl: 14 }
  // { 10: b@[2..3) lshl: 0 } { 11: b@[0..3) lshl: 1 } { 12: b@[0..3) lshl: 4 } { 13: b@[0..3) lshl: 7 } { 14: b@[0..3) lshl: 10 } { 15: b@[0..3) lshl: 13 }
  ///                                             0: b@[0..3), 5: b@[1..3), 10: b@[2..3)
  // CHECK-DAG: %[[MASK0:.*]] = arith.constant dense<[7, 6, 4]> : vector<3xi16>
  ///                                             1: b@[0..3), 6: b@[0..3), 11: b@[0..3)
  ///                                             ...
  // CHECK-DAG: %[[MASK1:.*]] = arith.constant dense<7> : vector<3xi16>
  ///                                             5: b@[0..1), 10: b@[0..2), 15: b@[0..3)
  // CHECK-DAG: %[[MASK2:.*]] = arith.constant dense<[1, 3, 7]> : vector<3xi16>
  // CHECK-DAG: %[[SHR0_CST:.*]] = arith.constant dense<[0, 1, 2]> : vector<3xi16>
  // CHECK-DAG: %[[SHL1_CST:.*]] = arith.constant dense<[3, 2, 1]> : vector<3xi16>
  // CHECK-DAG: %[[SHL2_CST:.*]] = arith.constant dense<[6, 5, 4]> : vector<3xi16>
  // CHECK-DAG: %[[SHL3_CST:.*]] = arith.constant dense<[9, 8, 7]> : vector<3xi16>
  // CHECK-DAG: %[[SHL4_CST:.*]] = arith.constant dense<[12, 11, 10]> : vector<3xi16>
  // CHECK-DAG: %[[SHL5_CST:.*]] = arith.constant dense<[15, 14, 13]> : vector<3xi16>

  //
  // CHECK: %[[V0:.*]] = vector.shuffle %[[A]], %[[A]] [0, 5, 10] : vector<16xi16>, vector<16xi16>
  // CHECK: %[[A0:.*]] = arith.andi %[[V0]], %[[MASK0]] : vector<3xi16>
  // CHECK: %[[SHR0:.*]] = arith.shrui %[[A0]], %[[SHR0_CST]] : vector<3xi16>
  // CHECK: %[[V1:.*]] = vector.shuffle %[[A]], %[[A]] [1, 6, 11] : vector<16xi16>, vector<16xi16>
  // CHECK: %[[A1:.*]] = arith.andi %[[V1]], %[[MASK1]] : vector<3xi16>
  // CHECK: %[[SHL1:.*]] = arith.shli %[[A1]], %[[SHL1_CST]] : vector<3xi16>
  // CHECK: %[[O1:.*]] = arith.ori %[[SHR0]], %[[SHL1]] : vector<3xi16>
  // CHECK: %[[V2:.*]] = vector.shuffle %[[A]], %[[A]] [2, 7, 12] : vector<16xi16>, vector<16xi16>
  // CHECK: %[[A2:.*]] = arith.andi %[[V2]], %[[MASK1]] : vector<3xi16>
  // CHECK: %[[SHL2:.*]] = arith.shli %[[A2]], %[[SHL2_CST]] : vector<3xi16>
  // CHECK: %[[O2:.*]] = arith.ori %[[O1]], %[[SHL2]] : vector<3xi16>
  // CHECK: %[[V3:.*]] = vector.shuffle %[[A]], %[[A]] [3, 8, 13] : vector<16xi16>, vector<16xi16>
  // CHECK: %[[A3:.*]] = arith.andi %[[V3]], %[[MASK1]] : vector<3xi16>
  // CHECK: %[[SHL3:.*]] = arith.shli %[[A3]], %[[SHL3_CST]] : vector<3xi16>
  // CHECK: %[[O3:.*]] = arith.ori %[[O2]], %[[SHL3]]  : vector<3xi16>
  // CHECK: %[[V4:.*]] = vector.shuffle %[[A]], %[[A]] [4, 9, 14] : vector<16xi16>, vector<16xi16>
  // CHECK: %[[A4:.*]] = arith.andi %[[V4]], %[[MASK1]] : vector<3xi16>
  // CHECK: %[[SHL4:.*]] = arith.shli %[[A4]], %[[SHL4_CST]] : vector<3xi16>
  // CHECK: %[[O4:.*]] = arith.ori %[[O3]], %[[SHL4]]  : vector<3xi16>
  // CHECK: %[[V5:.*]] = vector.shuffle %[[A]], %[[A]] [5, 10, 15] : vector<16xi16>, vector<16xi16>
  // CHECK: %[[A5:.*]] = arith.andi %[[V5]], %[[MASK2]] : vector<3xi16>
  // CHECK: %[[SHL5:.*]] = arith.shli %[[A5]], %[[SHL5_CST]] : vector<3xi16>
  // CHECK: %[[O5:.*]] = arith.ori %[[O4]], %[[SHL5]]  : vector<3xi16>
  /// No trunci needed as the result is already in i16.
  // CHECK-NOT: arith.trunci
  // CHECK-NOT: bitcast
  %0 = arith.trunci %a : vector<16xi16> to vector<16xi3>
  %1 = vector.bitcast %0 : vector<16xi3> to vector<3xi16>
  return %1 : vector<3xi16>
}

/// This pattern requires an extui 16 -> 32 and not a trunci.
// CHECK-LABEL: func.func @f3(
func.func @f3(%a: vector<16xi16>) -> vector<2xi32> {
  /// Rewriting this standalone pattern is about 25x faster on skylake-ax512 according to llvm-mca.
  /// Benefit further increases when mixed with other compute ops.
  ///
  // CHECK-NOT: arith.trunci
  // CHECK-NOT: bitcast
  //     CHECK: arith.extui
  %0 = arith.trunci %a : vector<16xi16> to vector<16xi4>
  %1 = vector.bitcast %0 : vector<16xi4> to vector<2xi32>
  return %1 : vector<2xi32>
}

/// This pattern is not rewritten as the result i6 is not a multiple of i8.
// CHECK-LABEL: func.func @f4(
func.func @f4(%a: vector<16xi16>) -> vector<8xi6> {
  // CHECK: trunci
  // CHECK: bitcast
  // CHECK-NOT: shuffle
  // CHECK-NOT: andi
  // CHECK-NOT: ori
  %0 = arith.trunci %a : vector<16xi16> to vector<16xi3>
  %1 = vector.bitcast %0 : vector<16xi3> to vector<8xi6>
  return %1 : vector<8xi6>
}

// CHECK-LABEL: func.func @f1ext(
//  CHECK-SAME: %[[A:[0-9a-z]*]]: vector<5xi8>) -> vector<8xi16> {
func.func @f1ext(%a: vector<5xi8>) -> vector<8xi16> {
  // CHECK-DAG: %[[MASK0:.*]] = arith.constant dense<[31, -32, 124, -128, -16, 62, -64, -8]> : vector<8xi8>
  // CHECK-DAG: %[[MASK1:.*]] = arith.constant dense<[0, 3, 0, 15, 1, 0, 7, 0]> : vector<8xi8>
  // CHECK-DAG: %[[SHR0_CST:.*]] = arith.constant dense<[0, 5, 2, 7, 4, 1, 6, 3]> : vector<8xi8>
  // CHECK-DAG: %[[SHL1_CST:.*]] = arith.constant dense<[5, 3, 5, 1, 4, 5, 2, 5]> : vector<8xi8>
  // CHECK: %[[V0:.*]] = vector.shuffle %[[A]], %[[A]] [0, 0, 1, 1, 2, 3, 3, 4] : vector<5xi8>, vector<5xi8>
  // CHECK: %[[A0:.*]] = arith.andi %[[V0]], %[[MASK0]] : vector<8xi8>
  // CHECK: %[[SHR0:.*]] = arith.shrui %[[A0]], %[[SHR0_CST]] : vector<8xi8>
  // CHECK: %[[V1:.*]] = vector.shuffle %[[A]], %[[A]] [0, 1, 0, 2, 3, 0, 4, 0] : vector<5xi8>, vector<5xi8>
  // CHECK: %[[A1:.*]] = arith.andi %[[V1]], %[[MASK1]] : vector<8xi8>
  // CHECK: %[[SHL1:.*]] = arith.shli %[[A1]], %[[SHL1_CST]] : vector<8xi8>
  // CHECK: %[[O1:.*]] = arith.ori %[[SHR0]], %[[SHL1]] : vector<8xi8>
  // CHECK: %[[RES:.*]] = arith.extsi %[[O1]] : vector<8xi8> to vector<8xi16>
  // return %[[RES]] : vector<8xi16>

  %0 = vector.bitcast %a : vector<5xi8> to vector<8xi5>
  %1 = arith.extsi %0 : vector<8xi5> to vector<8xi16>
  return %1 : vector<8xi16>
}

// CHECK-LABEL: func.func @f2ext(
//  CHECK-SAME: %[[A:[0-9a-z]*]]: vector<5xi8>) -> vector<8xi16> {
func.func @f2ext(%a: vector<5xi8>) -> vector<8xi16> {
  // CHECK-NOT: arith.extsi {{.*}} : vector<8xi8> to vector<8xi16>
  //     CHECK: %[[RES:.*]] = arith.extui {{.*}} : vector<8xi8> to vector<8xi16>
  // return %[[RES]] : vector<8xi16>

  %0 = vector.bitcast %a : vector<5xi8> to vector<8xi5>
  %1 = arith.extui %0 : vector<8xi5> to vector<8xi16>
  return %1 : vector<8xi16>
}

// CHECK-LABEL: func.func @f3ext(
//  CHECK-SAME: %[[A:[0-9a-z]*]]: vector<5xi8>) -> vector<8xi17> {
func.func @f3ext(%a: vector<5xi8>) -> vector<8xi17> {
  // CHECK: bitcast
  // CHECK: extsi
  // CHECK-NOT: shuffle
  // CHECK-NOT: andi
  // CHECK-NOT: ori
  %0 = vector.bitcast %a : vector<5xi8> to vector<8xi5>
  %1 = arith.extsi %0 : vector<8xi5> to vector<8xi17>
  return %1 : vector<8xi17>
}

// CHECK-LABEL: func.func @aligned_extsi(
func.func @aligned_extsi(%a: vector<8xi4>) -> vector<8xi32> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi4>) -> vector<8xi32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[SHL_LOW:.*]] = arith.shli %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.shrsi %[[SHL_LOW]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrsi %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
// CHECK:           %[[I32:.*]] = arith.extsi %[[INTERLEAVE]] : vector<8xi8> to vector<8xi32>
  %0 = arith.extsi %a : vector<8xi4> to vector<8xi32>
  return %0 : vector<8xi32>
}

// CHECK-LABEL: func.func @aligned_extsi_2d(
func.func @aligned_extsi_2d(%a: vector<8x32xi4>) -> vector<8x32xi32> {
// CHECK-SAME:    %[[IN:.*]]: vector<8x32xi4>) -> vector<8x32xi32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<8x16xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8x32xi4> to vector<8x16xi8>
// CHECK:           %[[SHL_LOW:.*]] = arith.shli %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[LOW:.*]] = arith.shrsi %[[SHL_LOW]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrsi %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<8x16xi8>
// CHECK:           %[[I32:.*]] = arith.extsi %[[INTERLEAVE]] : vector<8x32xi8> to vector<8x32xi32>
  %0 = arith.extsi %a : vector<8x32xi4> to vector<8x32xi32>
  return %0 : vector<8x32xi32>
}

// CHECK-LABEL: func.func @aligned_extsi_base_case(
func.func @aligned_extsi_base_case(%a: vector<8xi4>) -> vector<8xi8> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi4>) -> vector<8xi8> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[SHL_LOW:.*]] = arith.shli %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.shrsi %[[SHL_LOW]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrsi %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
  %0 = arith.extsi %a : vector<8xi4> to vector<8xi8>
  return %0 : vector<8xi8>
}

// CHECK-LABEL: func.func @aligned_sitofp(
func.func @aligned_sitofp(%a: vector<8xi4>) -> vector<8xf32> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi4>) -> vector<8xf32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[SHL_LOW:.*]] = arith.shli %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.shrsi %[[SHL_LOW]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrsi %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
// CHECK:           %[[F32:.*]] = arith.sitofp %[[INTERLEAVE]] : vector<8xi8> to vector<8xf32>
  %0 = arith.sitofp %a : vector<8xi4> to vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func.func @aligned_sitofp_2d(
func.func @aligned_sitofp_2d(%a: vector<8x32xi4>) -> vector<8x32xf32> {
// CHECK-SAME:    %[[IN:.*]]: vector<8x32xi4>) -> vector<8x32xf32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<8x16xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8x32xi4> to vector<8x16xi8>
// CHECK:           %[[SHL_LOW:.*]] = arith.shli %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[LOW:.*]] = arith.shrsi %[[SHL_LOW]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrsi %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<8x16xi8>
// CHECK:           %[[F32:.*]] = arith.sitofp %[[INTERLEAVE]] : vector<8x32xi8> to vector<8x32xf32>
  %0 = arith.sitofp %a : vector<8x32xi4> to vector<8x32xf32>
  return %0 : vector<8x32xf32>
}

// CHECK-LABEL: func.func @aligned_trunci(
func.func @aligned_trunci(%a: vector<8xi32>) -> vector<8xi4> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi32>) -> vector<8xi4> {
// CHECK-DAG:       %[[LOW_MASK:.*]] = arith.constant dense<15> : vector<4xi8>
// CHECK-DAG:       %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[I8:.*]] = arith.trunci %[[IN]] : vector<8xi32> to vector<8xi8>
// CHECK:           %[[LOW:.*]] = vector.shuffle %[[I8]], %[[I8]] [0, 2, 4, 6] : vector<8xi8>, vector<8xi8>
// CHECK:           %[[HIGH:.*]] = vector.shuffle %[[I8]], %[[I8]] [1, 3, 5, 7] : vector<8xi8>, vector<8xi8>
// CHECK:           %[[ZEROED_LOW:.*]] = arith.andi %[[LOW]], %[[LOW_MASK]] : vector<4xi8>
// CHECK:           %[[SHL_HIGH:.*]] = arith.shli %[[HIGH]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[MERGED:.*]] = arith.ori %[[ZEROED_LOW]], %[[SHL_HIGH]] : vector<4xi8>
// CHECK:           %[[I4:.*]] = vector.bitcast %[[MERGED]] : vector<4xi8> to vector<8xi4>
  %0 = arith.trunci %a : vector<8xi32> to vector<8xi4>
  return %0 : vector<8xi4>
}

// CHECK-LABEL: func.func @aligned_trunci_base_case(
func.func @aligned_trunci_base_case(%a: vector<8xi8>) -> vector<8xi4> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi8>) -> vector<8xi4> {
// CHECK-DAG:       %[[LOW_MASK:.*]] = arith.constant dense<15> : vector<4xi8>
// CHECK-DAG:       %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[LOW:.*]] = vector.shuffle %[[IN]], %[[IN]] [0, 2, 4, 6] : vector<8xi8>, vector<8xi8>
// CHECK:           %[[HIGH:.*]] = vector.shuffle %[[IN]], %[[IN]] [1, 3, 5, 7] : vector<8xi8>, vector<8xi8>
// CHECK:           %[[ZEROED_LOW:.*]] = arith.andi %[[LOW]], %[[LOW_MASK]] : vector<4xi8>
// CHECK:           %[[SHL_HIGH:.*]] = arith.shli %[[HIGH]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[MERGED:.*]] = arith.ori %[[ZEROED_LOW]], %[[SHL_HIGH]] : vector<4xi8>
// CHECK:           %[[I4:.*]] = vector.bitcast %[[MERGED]] : vector<4xi8> to vector<8xi4>
  %0 = arith.trunci %a : vector<8xi8> to vector<8xi4>
  return %0 : vector<8xi4>
}

// CHECK-LABEL: func.func @aligned_trunci_2d(
func.func @aligned_trunci_2d(%a: vector<8x32xi32>) -> vector<8x32xi4> {
// CHECK-NOT:       vector.shuffle
// CHECK-NOT:       vector.andi
// CHECK-NOT:       vector.shli
// CHECK-NOT:       vector.ori
// CHECK:           arith.trunci
  %0 = arith.trunci %a : vector<8x32xi32> to vector<8x32xi4>
  return %0 : vector<8x32xi4>
}

// CHECK-LABEL: func.func @i4_transpose(
func.func @i4_transpose(%a: vector<8x16xi4>) -> vector<16x8xi4> {
// CHECK-SAME:    %[[IN:.*]]: vector<8x16xi4>) -> vector<16x8xi4> {
// CHECK:           %[[EXT:.*]] = vector.interleave
// CHECK:           %[[TRANS:.*]] = vector.transpose %[[EXT]], [1, 0] : vector<8x16xi8> to vector<16x8xi8>
// CHECK:           %[[TRUNC:.*]] = arith.trunci %[[TRANS]] : vector<16x8xi8> to vector<16x8xi4>
  %0 = vector.transpose %a, [1, 0] : vector<8x16xi4> to vector<16x8xi4>
  return %0 : vector<16x8xi4>
}

// CHECK-LABEL: func.func @i7_transpose(
func.func @i7_transpose(%a: vector<8x16xi7>) -> vector<16x8xi7> {
// CHECK-SAME:    %[[IN:.*]]: vector<8x16xi7>) -> vector<16x8xi7> {
// CHECK:           %[[EXT:.*]] = arith.extsi %[[IN]] : vector<8x16xi7> to vector<8x16xi8>
// CHECK:           %[[TRANS:.*]] = vector.transpose %[[EXT]], [1, 0] : vector<8x16xi8> to vector<16x8xi8>
// CHECK:           %[[TRUNC:.*]] = arith.trunci %[[TRANS]] : vector<16x8xi8> to vector<16x8xi7>
  %0 = vector.transpose %a, [1, 0] : vector<8x16xi7> to vector<16x8xi7>
  return %0 : vector<16x8xi7>
}

// CHECK-LABEL: func.func @aligned_extui(
func.func @aligned_extui(%a: vector<8xi4>) -> vector<8xi32> {
// CHECK-SAME:                             %[[IN:.*]]: vector<8xi4>) -> vector<8xi32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<15> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrui %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
// CHECK:           %[[I32:.*]] = arith.extui %[[INTERLEAVE]] : vector<8xi8> to vector<8xi32>
  %0 = arith.extui %a : vector<8xi4> to vector<8xi32>
  return %0 : vector<8xi32>
}

// CHECK-LABEL: func.func @aligned_extui_2d(
func.func @aligned_extui_2d(%a: vector<8x32xi4>) -> vector<8x32xi32> {
// CHECK-SAME:                                %[[VAL_0:.*]]: vector<8x32xi4>) -> vector<8x32xi32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<8x16xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<15> : vector<8x16xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[VAL_0]] : vector<8x32xi4> to vector<8x16xi8>
// CHECK:           %[[LOW:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<8x16xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrui %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<8x16xi8>
// CHECK:           %[[I32:.*]] = arith.extui %[[INTERLEAVE]] : vector<8x32xi8> to vector<8x32xi32>
  %0 = arith.extui %a : vector<8x32xi4> to vector<8x32xi32>
  return %0 : vector<8x32xi32>
}

// CHECK-LABEL: func.func @aligned_extui_base_case(
func.func @aligned_extui_base_case(%a: vector<8xi4>) -> vector<8xi8> {
// CHECK-SAME:                             %[[IN:.*]]: vector<8xi4>) -> vector<8xi8> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<15> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrui %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
  %0 = arith.extui %a : vector<8xi4> to vector<8xi8>
  return %0 : vector<8xi8>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
        : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.rewrite_narrow_types
    } : !transform.any_op
    transform.yield
  }
}

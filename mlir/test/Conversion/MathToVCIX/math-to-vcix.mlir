// RUN: mlir-opt --split-input-file --verify-diagnostics --test-math-to-vcix %s | FileCheck %s

// CHECK-LABEL:   func.func @cos(
// CHECK-SAME:                   %[[VAL_0:.*]]: vector<[8]xf32>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i64) -> vector<[8]xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : i64
// CHECK:           %[[VAL_3:.*]] = "vcix.v.iv"(%[[VAL_0]], %[[VAL_2]]) <{imm = 0 : i32, opcode = 0 : i64}> : (vector<[8]xf32>, i64) -> vector<[8]xf32>
// CHECK:           return %[[VAL_3]] : vector<[8]xf32>
// CHECK:         }
func.func @cos(%a: vector<[8] x f32>, %rvl: i64) -> vector<[8] x f32> {
  %res = math.cos %a : vector<[8] x f32>
  return %res : vector<[8] x f32>
}

// -----

// CHECK-LABEL:   func.func @cos_req_legalization(
// CHECK-SAME:                                    %[[VAL_0:.*]]: vector<[32]xf32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: i64) -> vector<[32]xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<0.000000e+00> : vector<[32]xf32>
// CHECK:           %[[VAL_4:.*]] = vector.scalable.extract %[[VAL_0]][0] : vector<[16]xf32> from vector<[32]xf32>
// CHECK:           %[[VAL_5:.*]] = "vcix.v.iv"(%[[VAL_4]], %[[VAL_2]]) <{imm = 0 : i32, opcode = 0 : i64}> : (vector<[16]xf32>, i64) -> vector<[16]xf32>
// CHECK:           %[[VAL_6:.*]] = vector.scalable.insert %[[VAL_5]], %[[VAL_3]][0] : vector<[16]xf32> into vector<[32]xf32>
// CHECK:           %[[VAL_7:.*]] = vector.scalable.extract %[[VAL_0]][16] : vector<[16]xf32> from vector<[32]xf32>
// CHECK:           %[[VAL_8:.*]] = "vcix.v.iv"(%[[VAL_7]], %[[VAL_2]]) <{imm = 0 : i32, opcode = 0 : i64}> : (vector<[16]xf32>, i64) -> vector<[16]xf32>
// CHECK:           %[[VAL_9:.*]] = vector.scalable.insert %[[VAL_8]], %[[VAL_6]][16] : vector<[16]xf32> into vector<[32]xf32>
// CHECK:           return %[[VAL_9]] : vector<[32]xf32>
// CHECK:         }
func.func @cos_req_legalization(%a: vector<[32] x f32>, %rvl: i64) -> vector<[32] x f32> {
  %res = math.cos %a : vector<[32] x f32>
  return %res : vector<[32] x f32>
}

// -----

// CHECK-LABEL:   func.func @cos_fixed(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<8xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i64) -> vector<8xf32> {
// CHECK:           %[[VAL_2:.*]] = "vcix.v.iv"(%[[VAL_0]]) <{imm = 0 : i32, opcode = 0 : i64}> : (vector<8xf32>) -> vector<8xf32>
// CHECK:           return %[[VAL_2]] : vector<8xf32>
// CHECK:         }
func.func @cos_fixed(%a: vector<8 x f32>, %rvl: i64) -> vector<8 x f32> {
  %res = math.cos %a : vector<8 x f32>
  return %res : vector<8 x f32>
}

// -----

// CHECK-LABEL:   func.func @sin(
// CHECK-SAME:                   %[[VAL_0:.*]]: vector<[8]xf32>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i64) -> vector<[8]xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : i64
// CHECK:           %[[VAL_3:.*]] = "vcix.v.sv"(%[[VAL_0]], %[[VAL_0]], %[[VAL_2]]) <{opcode = 0 : i64}> : (vector<[8]xf32>, vector<[8]xf32>, i64) -> vector<[8]xf32>
// CHECK:           return %[[VAL_3]] : vector<[8]xf32>
// CHECK:         }
func.func @sin(%a: vector<[8] x f32>, %rvl: i64) -> vector<[8] x f32> {
  %res = math.sin %a : vector<[8] x f32>
  return %res : vector<[8] x f32>
}

// -----

// CHECK-LABEL:   func.func @sin_req_legalization(
// CHECK-SAME:                                    %[[VAL_0:.*]]: vector<[32]xf32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: i64) -> vector<[32]xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<0.000000e+00> : vector<[32]xf32>
// CHECK:           %[[VAL_4:.*]] = vector.scalable.extract %[[VAL_0]][0] : vector<[16]xf32> from vector<[32]xf32>
// CHECK:           %[[VAL_5:.*]] = "vcix.v.sv"(%[[VAL_4]], %[[VAL_4]], %[[VAL_2]]) <{opcode = 0 : i64}> : (vector<[16]xf32>, vector<[16]xf32>, i64) -> vector<[16]xf32>
// CHECK:           %[[VAL_6:.*]] = vector.scalable.insert %[[VAL_5]], %[[VAL_3]][0] : vector<[16]xf32> into vector<[32]xf32>
// CHECK:           %[[VAL_7:.*]] = vector.scalable.extract %[[VAL_0]][16] : vector<[16]xf32> from vector<[32]xf32>
// CHECK:           %[[VAL_8:.*]] = "vcix.v.sv"(%[[VAL_7]], %[[VAL_7]], %[[VAL_2]]) <{opcode = 0 : i64}> : (vector<[16]xf32>, vector<[16]xf32>, i64) -> vector<[16]xf32>
// CHECK:           %[[VAL_9:.*]] = vector.scalable.insert %[[VAL_8]], %[[VAL_6]][16] : vector<[16]xf32> into vector<[32]xf32>
// CHECK:           return %[[VAL_9]] : vector<[32]xf32>
// CHECK:         }
func.func @sin_req_legalization(%a: vector<[32] x f32>, %rvl: i64) -> vector<[32] x f32> {
  %res = math.sin %a : vector<[32] x f32>
  return %res : vector<[32] x f32>
}

// -----

// CHECK-LABEL:   func.func @sin_fixed(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<8xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i64) -> vector<8xf32> {
// CHECK:           %[[VAL_2:.*]] = "vcix.v.sv"(%[[VAL_0]], %[[VAL_0]]) <{opcode = 0 : i64}> : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK:           return %[[VAL_2]] : vector<8xf32>
// CHECK:         }
func.func @sin_fixed(%a: vector<8 x f32>, %rvl: i64) -> vector<8 x f32> {
  %res = math.sin %a : vector<8 x f32>
  return %res : vector<8 x f32>
}

// -----

// CHECK-LABEL:   func.func @tan(
// CHECK-SAME:                   %[[VAL_0:.*]]: vector<[8]xf32>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i64) -> vector<[8]xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 9 : i64
// CHECK:           %[[VAL_4:.*]] = "vcix.v.sv"(%[[VAL_0]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 0 : i64}> : (vector<[8]xf32>, f32, i64) -> vector<[8]xf32>
// CHECK:           return %[[VAL_4]] : vector<[8]xf32>
// CHECK:         }
func.func @tan(%a: vector<[8] x f32>, %rvl: i64) -> vector<[8] x f32> {
  %res = math.tan %a : vector<[8] x f32>
  return %res : vector<[8] x f32>
}

// -----

// CHECK-LABEL:   func.func @tan_req_legalization(
// CHECK-SAME:                                    %[[VAL_0:.*]]: vector<[32]xf32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: i64) -> vector<[32]xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 9 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<0.000000e+00> : vector<[32]xf32>
// CHECK:           %[[VAL_5:.*]] = vector.scalable.extract %[[VAL_0]][0] : vector<[16]xf32> from vector<[32]xf32>
// CHECK:           %[[VAL_6:.*]] = "vcix.v.sv"(%[[VAL_5]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 0 : i64}> : (vector<[16]xf32>, f32, i64) -> vector<[16]xf32>
// CHECK:           %[[VAL_7:.*]] = vector.scalable.insert %[[VAL_6]], %[[VAL_4]][0] : vector<[16]xf32> into vector<[32]xf32>
// CHECK:           %[[VAL_8:.*]] = vector.scalable.extract %[[VAL_0]][16] : vector<[16]xf32> from vector<[32]xf32>
// CHECK:           %[[VAL_9:.*]] = "vcix.v.sv"(%[[VAL_8]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 0 : i64}> : (vector<[16]xf32>, f32, i64) -> vector<[16]xf32>
// CHECK:           %[[VAL_10:.*]] = vector.scalable.insert %[[VAL_9]], %[[VAL_7]][16] : vector<[16]xf32> into vector<[32]xf32>
// CHECK:           return %[[VAL_10]] : vector<[32]xf32>
// CHECK:         }
func.func @tan_req_legalization(%a: vector<[32] x f32>, %rvl: i64) -> vector<[32] x f32> {
  %res = math.tan %a : vector<[32] x f32>
  return %res : vector<[32] x f32>
}

// -----

// CHECK-LABEL:   func.func @tan_fixed(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<8xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i64) -> vector<8xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = "vcix.v.sv"(%[[VAL_0]], %[[VAL_2]]) <{opcode = 0 : i64}> : (vector<8xf32>, f32) -> vector<8xf32>
// CHECK:           return %[[VAL_3]] : vector<8xf32>
// CHECK:         }
func.func @tan_fixed(%a: vector<8 x f32>, %rvl: i64) -> vector<8 x f32> {
  %res = math.tan %a : vector<8 x f32>
  return %res : vector<8 x f32>
}

// -----

// CHECK-LABEL:   func.func @log(
// CHECK-SAME:                   %[[VAL_0:.*]]: vector<[8]xf32>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i64) -> vector<[8]xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 9 : i64
// CHECK:           %[[VAL_4:.*]] = "vcix.v.sv"(%[[VAL_0]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 0 : i64}> : (vector<[8]xf32>, i32, i64) -> vector<[8]xf32>
// CHECK:           return %[[VAL_4]] : vector<[8]xf32>
// CHECK:         }
func.func @log(%a: vector<[8] x f32>, %rvl: i64) -> vector<[8] x f32> {
  %res = math.log %a : vector<[8] x f32>
  return %res : vector<[8] x f32>
}

// -----

// CHECK-LABEL:   func.func @log_req_legalization(
// CHECK-SAME:                                    %[[VAL_0:.*]]: vector<[32]xf32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: i64) -> vector<[32]xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 9 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<0.000000e+00> : vector<[32]xf32>
// CHECK:           %[[VAL_5:.*]] = vector.scalable.extract %[[VAL_0]][0] : vector<[16]xf32> from vector<[32]xf32>
// CHECK:           %[[VAL_6:.*]] = "vcix.v.sv"(%[[VAL_5]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 0 : i64}> : (vector<[16]xf32>, i32, i64) -> vector<[16]xf32>
// CHECK:           %[[VAL_7:.*]] = vector.scalable.insert %[[VAL_6]], %[[VAL_4]][0] : vector<[16]xf32> into vector<[32]xf32>
// CHECK:           %[[VAL_8:.*]] = vector.scalable.extract %[[VAL_0]][16] : vector<[16]xf32> from vector<[32]xf32>
// CHECK:           %[[VAL_9:.*]] = "vcix.v.sv"(%[[VAL_8]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 0 : i64}> : (vector<[16]xf32>, i32, i64) -> vector<[16]xf32>
// CHECK:           %[[VAL_10:.*]] = vector.scalable.insert %[[VAL_9]], %[[VAL_7]][16] : vector<[16]xf32> into vector<[32]xf32>
// CHECK:           return %[[VAL_10]] : vector<[32]xf32>
// CHECK:         }
func.func @log_req_legalization(%a: vector<[32] x f32>, %rvl: i64) -> vector<[32] x f32> {
  %res = math.log %a : vector<[32] x f32>
  return %res : vector<[32] x f32>
}

// -----

// CHECK-LABEL:   func.func @log_fixed(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<8xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i64) -> vector<8xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = "vcix.v.sv"(%[[VAL_0]], %[[VAL_2]]) <{opcode = 0 : i64}> : (vector<8xf32>, i32) -> vector<8xf32>
// CHECK:           return %[[VAL_3]] : vector<8xf32>
// CHECK:         }
func.func @log_fixed(%a: vector<8 x f32>, %rvl: i64) -> vector<8 x f32> {
  %res = math.log %a : vector<8 x f32>
  return %res : vector<8 x f32>
}

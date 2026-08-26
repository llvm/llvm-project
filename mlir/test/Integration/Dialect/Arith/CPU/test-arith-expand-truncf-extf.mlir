// Check various edge cases for truncf/extf ops involving f32 and the f4E2M1FN,
// f8E4M3FN and f8E5M2 types.

// RUN: mlir-opt %s --convert-func-to-llvm \
// RUN:             --arith-expand="include-f4e2m1=true include-f8e4m3fn=true include-f8e5m2=true" \
// RUN:             --convert-arith-to-llvm --convert-vector-to-llvm \
// RUN:             --reconcile-unrealized-casts | \
// RUN:   mlir-runner -e entry --entry-point-result=void \
// RUN:               --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @check_extf(%in : f4E2M1FN) -> () {
  %res = arith.extf %in : f4E2M1FN to f32
  vector.print %res : f32
  return
}

// See https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
// for details on F4E2M1 representation 
func.func @check_truncf(%in : f32) -> () {
  %trunc = arith.truncf %in : f32 to f4E2M1FN
  %bitcast = arith.bitcast %trunc : f4E2M1FN to i4
  %res = arith.extui %bitcast : i4 to i64
  vector.print %res : i64
  return
}

func.func @check_extf_f8E4M3FN(%in : f8E4M3FN) -> () {
  %res = arith.extf %in : f8E4M3FN to f32
  vector.print %res : f32
  return
}

// F8E4M3FN has no infinity; the maximum representable magnitude is 448 and
// 0x7f/0xff are the only NaN encodings, so overflow (and infinity/NaN inputs)
// map to NaN rather than saturating.
func.func @check_truncf_f8E4M3FN(%in : f32) -> () {
  %trunc = arith.truncf %in : f32 to f8E4M3FN
  %bitcast = arith.bitcast %trunc : f8E4M3FN to i8
  %res = arith.extui %bitcast : i8 to i64
  vector.print %res : i64
  return
}

func.func @check_extf_f8E5M2(%in : f8E5M2) -> () {
  %res = arith.extf %in : f8E5M2 to f32
  vector.print %res : f32
  return
}

// F8E5M2 is IEEE-like with infinities (0x7c/0xfc) and NaNs, so an overflowing
// magnitude rounds to infinity rather than saturating or becoming NaN.
func.func @check_truncf_f8E5M2(%in : f32) -> () {
  %trunc = arith.truncf %in : f32 to f8E5M2
  %bitcast = arith.bitcast %trunc : f8E5M2 to i8
  %res = arith.extui %bitcast : i8 to i64
  vector.print %res : i64
  return
}

func.func @entry() {
  %zero = arith.constant 0.0 : f32
  %half = arith.constant 0.5 : f32
  %one = arith.constant 1.0 : f32
  %oneAndAHalf = arith.constant 1.5 : f32
  %two = arith.constant 2.0 : f32
  %three = arith.constant 3.0 : f32
  %four = arith.constant 4.0 : f32
  %max = arith.constant 6.0 : f32
  %minZero = arith.constant -0.0 : f32
  %minHalf = arith.constant -0.5 : f32
  %minOne = arith.constant -1.0 : f32
  %minOneAndAHalf = arith.constant -1.5 : f32
  %minTwo = arith.constant -2.0 : f32
  %minThree = arith.constant -3.0 : f32
  %minFour = arith.constant -4.0 : f32
  %min = arith.constant -6.0 : f32
  %lowerThanMin = arith.constant -1000000.0 : f32
  %higherThanMax = arith.constant 1000000.0 : f32
  %mustRound = arith.constant -3.14 : f32
  %nan = arith.constant 0x7f80000 : f32

  // CHECK: 0
  func.call @check_truncf(%zero) : (f32) -> ()
  // CHECK: 1
  func.call @check_truncf(%half) : (f32) -> ()
  // CHECK: 2
  func.call @check_truncf(%one) : (f32) -> ()
  // CHECK: 3
  func.call @check_truncf(%oneAndAHalf) : (f32) -> ()
  // CHECK: 4
  func.call @check_truncf(%two) : (f32) -> ()
  // CHECK: 5
  func.call @check_truncf(%three) : (f32) -> ()
  // CHECK: 6
  func.call @check_truncf(%four) : (f32) -> ()
  // CHECK: 7
  func.call @check_truncf(%max) : (f32) -> ()
  // CHECK: 9
  func.call @check_truncf(%minHalf) : (f32) -> ()
  // CHECK: 10
  func.call @check_truncf(%minOne) : (f32) -> ()
  // CHECK: 11
  func.call @check_truncf(%minOneAndAHalf) : (f32) -> ()
  // CHECK: 12
  func.call @check_truncf(%minTwo) : (f32) -> ()
  // CHECK: 13
  func.call @check_truncf(%minThree) : (f32) -> ()
  // CHECK: 14
  func.call @check_truncf(%minFour) : (f32) -> ()
  // CHECK: 15
  func.call @check_truncf(%min) : (f32) -> ()
  // CHECK: 7
  func.call @check_truncf(%higherThanMax) : (f32) -> ()
  // CHECK: 15
  func.call @check_truncf(%lowerThanMin) : (f32) -> ()
  // CHECK: 13
  func.call @check_truncf(%mustRound) : (f32) -> ()
  // CHECK: 0
  func.call @check_truncf(%nan) : (f32) -> ()

  // CHECK: 0
  %zeroF4 = arith.truncf %zero : f32 to f4E2M1FN
  func.call @check_extf(%zeroF4) : (f4E2M1FN) -> ()
  // CHECK: 0.5
  %halfF4 = arith.truncf %half : f32 to f4E2M1FN
  func.call @check_extf(%halfF4) : (f4E2M1FN) -> ()
  // CHECK: 1
  %oneF4 = arith.truncf %one : f32 to f4E2M1FN
  func.call @check_extf(%oneF4) : (f4E2M1FN) -> ()
  // CHECK: 1.5
  %oneAndAHalfF4 = arith.truncf %oneAndAHalf : f32 to f4E2M1FN
  func.call @check_extf(%oneAndAHalfF4) : (f4E2M1FN) -> ()
  // CHECK: 2
  %twoF4 = arith.truncf %two : f32 to f4E2M1FN
  func.call @check_extf(%twoF4) : (f4E2M1FN) -> ()
  // CHECK: 3
  %threeF4 = arith.truncf %three : f32 to f4E2M1FN
  func.call @check_extf(%threeF4) : (f4E2M1FN) -> ()
  // CHECK: 4
  %fourF4 = arith.truncf %four : f32 to f4E2M1FN
  func.call @check_extf(%fourF4) : (f4E2M1FN) -> ()
  // CHECK: 6
  %higherThanMaxF4 = arith.truncf %higherThanMax : f32 to f4E2M1FN
  func.call @check_extf(%higherThanMaxF4) : (f4E2M1FN) -> ()
  // CHECK: -0
  %minZeroF4 = arith.truncf %minZero : f32 to f4E2M1FN
  func.call @check_extf(%minZeroF4) : (f4E2M1FN) -> ()
  // CHECK: -0.5
  %minHalfF4 = arith.truncf %minHalf : f32 to f4E2M1FN
  func.call @check_extf(%minHalfF4) : (f4E2M1FN) -> ()
  // CHECK: -1
  %minOneF4 = arith.truncf %minOne : f32 to f4E2M1FN
  func.call @check_extf(%minOneF4) : (f4E2M1FN) -> ()
  // CHECK: -1.5
  %minOneAndAHalfF4 = arith.truncf %minOneAndAHalf : f32 to f4E2M1FN
  func.call @check_extf(%minOneAndAHalfF4) : (f4E2M1FN) -> ()
  // CHECK: -2
  %minTwoF4 = arith.truncf %minTwo : f32 to f4E2M1FN
  func.call @check_extf(%minTwoF4) : (f4E2M1FN) -> ()
  // CHECK: -3
  %minThreeF4 = arith.truncf %minThree : f32 to f4E2M1FN
  func.call @check_extf(%minThreeF4) : (f4E2M1FN) -> ()
  // CHECK: -4
  %minFourF4 = arith.truncf %minFour : f32 to f4E2M1FN
  func.call @check_extf(%minFourF4) : (f4E2M1FN) -> ()
  // CHECK: -6
  %lowerThanMinF4 = arith.truncf %lowerThanMin : f32 to f4E2M1FN
  func.call @check_extf(%lowerThanMinF4) : (f4E2M1FN) -> ()
  // CHECK: -3
  %mustRoundF4 = arith.truncf %mustRound : f32 to f4E2M1FN
  func.call @check_extf(%mustRoundF4) : (f4E2M1FN) -> ()

  // F8E4M3FN checks. See the OCP FP8 (E4M3) spec for the representation.
  %e4m3Round = arith.constant 1.1 : f32
  %e4m3Max = arith.constant 448.0 : f32
  %e4m3Tie = arith.constant 464.0 : f32
  %e4m3Ovf = arith.constant 465.0 : f32
  %e4m3Min = arith.constant -448.0 : f32
  %qnan = arith.constant 0x7fc00000 : f32

  // CHECK: 0
  func.call @check_truncf_f8E4M3FN(%zero) : (f32) -> ()
  // CHECK: 48
  func.call @check_truncf_f8E4M3FN(%half) : (f32) -> ()
  // CHECK: 56
  func.call @check_truncf_f8E4M3FN(%one) : (f32) -> ()
  // CHECK: 60
  func.call @check_truncf_f8E4M3FN(%oneAndAHalf) : (f32) -> ()
  // CHECK: 64
  func.call @check_truncf_f8E4M3FN(%two) : (f32) -> ()
  // CHECK: 68
  func.call @check_truncf_f8E4M3FN(%three) : (f32) -> ()
  // CHECK: 72
  func.call @check_truncf_f8E4M3FN(%four) : (f32) -> ()
  // CHECK: 57
  func.call @check_truncf_f8E4M3FN(%e4m3Round) : (f32) -> ()
  // CHECK: 126
  func.call @check_truncf_f8E4M3FN(%e4m3Max) : (f32) -> ()
  // A magnitude exactly halfway between the max (448) and the NaN slot rounds
  // to even, i.e. to the finite max.
  // CHECK: 126
  func.call @check_truncf_f8E4M3FN(%e4m3Tie) : (f32) -> ()
  // Just past the tie, the value overflows and maps to NaN (0x7f).
  // CHECK: 127
  func.call @check_truncf_f8E4M3FN(%e4m3Ovf) : (f32) -> ()
  // CHECK: 127
  func.call @check_truncf_f8E4M3FN(%higherThanMax) : (f32) -> ()
  // CHECK: 176
  func.call @check_truncf_f8E4M3FN(%minHalf) : (f32) -> ()
  // CHECK: 184
  func.call @check_truncf_f8E4M3FN(%minOne) : (f32) -> ()
  // CHECK: 200
  func.call @check_truncf_f8E4M3FN(%minFour) : (f32) -> ()
  // CHECK: 254
  func.call @check_truncf_f8E4M3FN(%e4m3Min) : (f32) -> ()
  // CHECK: 127
  func.call @check_truncf_f8E4M3FN(%lowerThanMin) : (f32) -> ()
  // CHECK: 127
  func.call @check_truncf_f8E4M3FN(%qnan) : (f32) -> ()

  // CHECK: 0.5
  %halfE4m3 = arith.truncf %half : f32 to f8E4M3FN
  func.call @check_extf_f8E4M3FN(%halfE4m3) : (f8E4M3FN) -> ()
  // CHECK: 1
  %oneE4m3 = arith.truncf %one : f32 to f8E4M3FN
  func.call @check_extf_f8E4M3FN(%oneE4m3) : (f8E4M3FN) -> ()
  // CHECK: 1.5
  %oneAndAHalfE4m3 = arith.truncf %oneAndAHalf : f32 to f8E4M3FN
  func.call @check_extf_f8E4M3FN(%oneAndAHalfE4m3) : (f8E4M3FN) -> ()
  // CHECK: 1.125
  %roundE4m3 = arith.truncf %e4m3Round : f32 to f8E4M3FN
  func.call @check_extf_f8E4M3FN(%roundE4m3) : (f8E4M3FN) -> ()
  // CHECK: 448
  %maxE4m3 = arith.truncf %e4m3Max : f32 to f8E4M3FN
  func.call @check_extf_f8E4M3FN(%maxE4m3) : (f8E4M3FN) -> ()
  // CHECK: 448
  %tieE4m3 = arith.truncf %e4m3Tie : f32 to f8E4M3FN
  func.call @check_extf_f8E4M3FN(%tieE4m3) : (f8E4M3FN) -> ()
  // CHECK: nan
  %ovfE4m3 = arith.truncf %e4m3Ovf : f32 to f8E4M3FN
  func.call @check_extf_f8E4M3FN(%ovfE4m3) : (f8E4M3FN) -> ()
  // CHECK: nan
  %higherE4m3 = arith.truncf %higherThanMax : f32 to f8E4M3FN
  func.call @check_extf_f8E4M3FN(%higherE4m3) : (f8E4M3FN) -> ()
  // CHECK: -4
  %minFourE4m3 = arith.truncf %minFour : f32 to f8E4M3FN
  func.call @check_extf_f8E4M3FN(%minFourE4m3) : (f8E4M3FN) -> ()
  // CHECK: nan
  %lowerE4m3 = arith.truncf %lowerThanMin : f32 to f8E4M3FN
  func.call @check_extf_f8E4M3FN(%lowerE4m3) : (f8E4M3FN) -> ()

  // F8E5M2 checks. See the OCP FP8 (E5M2) spec for the representation.
  %e5m2Round = arith.constant 1.2 : f32
  %e5m2Max = arith.constant 57344.0 : f32
  %e5m2Min = arith.constant -57344.0 : f32
  %inf = arith.constant 0x7f800000 : f32

  // CHECK: 0
  func.call @check_truncf_f8E5M2(%zero) : (f32) -> ()
  // CHECK: 56
  func.call @check_truncf_f8E5M2(%half) : (f32) -> ()
  // CHECK: 60
  func.call @check_truncf_f8E5M2(%one) : (f32) -> ()
  // CHECK: 61
  func.call @check_truncf_f8E5M2(%e5m2Round) : (f32) -> ()
  // CHECK: 62
  func.call @check_truncf_f8E5M2(%oneAndAHalf) : (f32) -> ()
  // CHECK: 64
  func.call @check_truncf_f8E5M2(%two) : (f32) -> ()
  // CHECK: 68
  func.call @check_truncf_f8E5M2(%four) : (f32) -> ()
  // CHECK: 123
  func.call @check_truncf_f8E5M2(%e5m2Max) : (f32) -> ()
  // Overflow rounds to positive infinity (0x7c) since F8E5M2 has infinities.
  // CHECK: 124
  func.call @check_truncf_f8E5M2(%higherThanMax) : (f32) -> ()
  // CHECK: 184
  func.call @check_truncf_f8E5M2(%minHalf) : (f32) -> ()
  // CHECK: 188
  func.call @check_truncf_f8E5M2(%minOne) : (f32) -> ()
  // CHECK: 196
  func.call @check_truncf_f8E5M2(%minFour) : (f32) -> ()
  // CHECK: 251
  func.call @check_truncf_f8E5M2(%e5m2Min) : (f32) -> ()
  // Negative overflow rounds to negative infinity (0xfc).
  // CHECK: 252
  func.call @check_truncf_f8E5M2(%lowerThanMin) : (f32) -> ()
  // CHECK: 124
  func.call @check_truncf_f8E5M2(%inf) : (f32) -> ()
  // CHECK: 126
  func.call @check_truncf_f8E5M2(%qnan) : (f32) -> ()

  // CHECK: 0.5
  %halfE5m2 = arith.truncf %half : f32 to f8E5M2
  func.call @check_extf_f8E5M2(%halfE5m2) : (f8E5M2) -> ()
  // CHECK: 1
  %oneE5m2 = arith.truncf %one : f32 to f8E5M2
  func.call @check_extf_f8E5M2(%oneE5m2) : (f8E5M2) -> ()
  // CHECK: 1.25
  %roundE5m2 = arith.truncf %e5m2Round : f32 to f8E5M2
  func.call @check_extf_f8E5M2(%roundE5m2) : (f8E5M2) -> ()
  // CHECK: 1.5
  %oneAndAHalfE5m2 = arith.truncf %oneAndAHalf : f32 to f8E5M2
  func.call @check_extf_f8E5M2(%oneAndAHalfE5m2) : (f8E5M2) -> ()
  // CHECK: 2
  %twoE5m2 = arith.truncf %two : f32 to f8E5M2
  func.call @check_extf_f8E5M2(%twoE5m2) : (f8E5M2) -> ()
  // CHECK: 57344
  %maxE5m2 = arith.truncf %e5m2Max : f32 to f8E5M2
  func.call @check_extf_f8E5M2(%maxE5m2) : (f8E5M2) -> ()
  // CHECK: inf
  %higherE5m2 = arith.truncf %higherThanMax : f32 to f8E5M2
  func.call @check_extf_f8E5M2(%higherE5m2) : (f8E5M2) -> ()
  // CHECK: -4
  %minFourE5m2 = arith.truncf %minFour : f32 to f8E5M2
  func.call @check_extf_f8E5M2(%minFourE5m2) : (f8E5M2) -> ()
  // CHECK: -inf
  %lowerE5m2 = arith.truncf %lowerThanMin : f32 to f8E5M2
  func.call @check_extf_f8E5M2(%lowerE5m2) : (f8E5M2) -> ()
  return
}

// Check various edge cases for truncf/extf ops involving f32 and f4e2m1 types.

// RUN: mlir-opt %s --convert-func-to-llvm \
// RUN:             --arith-expand="include-f4e2m1=true" \
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
  return
}

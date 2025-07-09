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
  %max = arith.constant 6.0 : f32
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
  // CHECK: 7
  func.call @check_truncf(%max) : (f32) -> ()
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
  // CHECK: 6
  %higherThanMaxF4 = arith.truncf %higherThanMax : f32 to f4E2M1FN
  func.call @check_extf(%higherThanMaxF4) : (f4E2M1FN) -> ()
  // CHECK: -6
  %lowerThanMinF4 = arith.truncf %lowerThanMin : f32 to f4E2M1FN
  func.call @check_extf(%lowerThanMinF4) : (f4E2M1FN) -> ()
  // CHECK: -3
  %mustRoundF4 = arith.truncf %mustRound : f32 to f4E2M1FN
  func.call @check_extf(%mustRoundF4) : (f4E2M1FN) -> ()
  return
}

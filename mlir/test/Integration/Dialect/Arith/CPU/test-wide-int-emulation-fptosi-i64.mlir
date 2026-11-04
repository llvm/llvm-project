// Check that the wide integer `arith.fptosi` emulation produces the same result as wide
// `arith.fptosi`. Emulate i64 ops with i32 ops.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// RUN: mlir-opt %s --test-arith-emulate-wide-int="widest-int-supported=32" \
// RUN:             --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// Ops in this function *only* will be emulated using i32 types.
func.func @emulate_fptosi(%arg: f64) -> i64 {
  %res = arith.fptosi %arg : f64 to i64
  return %res : i64
}

func.func @check_fptosi(%arg : f64) -> () {
  %res = func.call @emulate_fptosi(%arg) : (f64) -> (i64)
  vector.print %res : i64
  return
}

func.func @entry() {
  %cst0 = arith.constant 0.0 : f64
  %cst_nzero = arith.constant 0x8000000000000000 : f64
  %cst1 = arith.constant 1.0 : f64
  %cst_n1 = arith.constant -1.0 : f64
  %cst_n1_5 = arith.constant -1.5 : f64

  %cstpow20 = arith.constant 1048576.0 : f64
  %cstnpow20 = arith.constant -1048576.0 : f64

  %cst_i32_max = arith.constant 4294967295.0 : f64
  %cst_i32_min = arith.constant -4294967296.0 : f64
  %cst_i32_overflow = arith.constant 4294967296.0 : f64
  %cst_i32_noverflow = arith.constant -4294967297.0 : f64

  %cstpow40 = arith.constant 1099511627776.0 : f64
  %cstnpow40 = arith.constant -1099511627776.0 : f64
  %cst_pow40ppow20 = arith.constant 1099512676352.0 : f64
  %cst_npow40ppow20 = arith.constant -1099512676352.0 : f64

  %cst_max = arith.constant 9007199254740992.0
  %cst_min = arith.constant -9007199254740992.0

  // CHECK:         0
  func.call @check_fptosi(%cst0) : (f64) -> ()
  // CHECK-NEXT:    0
  func.call @check_fptosi(%cst_nzero) : (f64) -> ()
  // CHECK-NEXT:    1
  func.call @check_fptosi(%cst1) : (f64) -> ()
  // CHECK-NEXT:    -1
  func.call @check_fptosi(%cst_n1) : (f64) -> ()
  // CHECK-NEXT:    -1
  func.call @check_fptosi(%cst_n1_5) : (f64) -> ()
  // CHECK-NEXT:    1048576
  func.call @check_fptosi(%cstpow20) : (f64) -> ()
  // CHECK-NEXT:    -1048576
  func.call @check_fptosi(%cstnpow20) : (f64) -> ()
  // CHECK-NEXT:    4294967295
  func.call @check_fptosi(%cst_i32_max) : (f64) -> ()
  // CHECK-NEXT:    -4294967296
  func.call @check_fptosi(%cst_i32_min) : (f64) -> ()
  // CHECK-NEXT:    4294967296
  func.call @check_fptosi(%cst_i32_overflow) : (f64) -> ()
  // CHECK-NEXT:    -4294967297
  func.call @check_fptosi(%cst_i32_noverflow) : (f64) -> ()
  // CHECK-NEXT:    1099511627776
  func.call @check_fptosi(%cstpow40) : (f64) -> ()
  // CHECK-NEXT:    -1099511627776
  func.call @check_fptosi(%cstnpow40) : (f64) -> ()
  // CHECK-NEXT:    1099512676352
  func.call @check_fptosi(%cst_pow40ppow20) : (f64) -> ()
  // CHECK-NEXT:    -1099512676352
  func.call @check_fptosi(%cst_npow40ppow20) : (f64) -> ()
  // CHECK-NEXT:    9007199254740992
  func.call @check_fptosi(%cst_max) : (f64) -> ()
  // CHECK-NEXT:    -9007199254740992
  func.call @check_fptosi(%cst_min) : (f64) -> ()

  return
}

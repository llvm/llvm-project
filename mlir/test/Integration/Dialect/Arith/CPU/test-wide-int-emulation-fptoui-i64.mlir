// Check that the wide integer `arith.fptoui` emulation produces the same result as wide
// `arith.fptoui`. Emulate i64 ops with i32 ops.

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
func.func @emulate_fptoui(%arg: f64) -> i64 {
  %res = arith.fptoui %arg : f64 to i64
  return %res : i64
}

func.func @check_fptoui(%arg : f64) -> () {
  %res = func.call @emulate_fptoui(%arg) : (f64) -> (i64)
  vector.print %res : i64
  return
}

func.func @entry() {
  %cst0 = arith.constant 0.0 : f64
  %cst1 = arith.constant 1.0 : f64
  %cst1_5 = arith.constant 1.5 : f64

  %cstpow20 = arith.constant 1048576.0 : f64
  %cst_i32_max = arith.constant 4294967295.0 : f64
  %cst_i32_overflow = arith.constant 4294967296.0 : f64


  %cstpow40 = arith.constant 1099511627776.0 : f64
  %cst_pow40ppow20 = arith.constant 1099512676352.0 : f64

  %cst_nzero = arith.constant 0x8000000000000000 : f64

  // CHECK:         0
  func.call @check_fptoui(%cst0) : (f64) -> ()
  // CHECK-NEXT:    1
  func.call @check_fptoui(%cst1) : (f64) -> ()
  // CHECK-NEXT:    1
  func.call @check_fptoui(%cst1_5) : (f64) -> ()
  // CHECK-NEXT:    1048576
  func.call @check_fptoui(%cstpow20) : (f64) -> ()
  // CHECK-NEXT:    4294967295
  func.call @check_fptoui(%cst_i32_max) : (f64) -> ()
  // CHECK-NEXT:    4294967296
  func.call @check_fptoui(%cst_i32_overflow) : (f64) -> ()
  // CHECK-NEXT:    1099511627776
  func.call @check_fptoui(%cstpow40) : (f64) -> ()
  // CHECK-NEXT:    1099512676352
  func.call @check_fptoui(%cst_pow40ppow20) : (f64) -> ()
  // CHECK-NEXT:    0
  func.call @check_fptoui(%cst_nzero) : (f64) -> ()

  return
}

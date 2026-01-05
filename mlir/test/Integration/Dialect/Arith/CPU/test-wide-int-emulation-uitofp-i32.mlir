// Check that the wide integer `arith.uitofp` emulation produces the same result as wide
// `arith.uitofp`. Emulate i32 ops with i16 ops.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// RUN: mlir-opt %s --test-arith-emulate-wide-int="widest-int-supported=16" \
// RUN:             --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// Ops in this function *only* will be emulated using i16 types.
func.func @emulate_uitofp(%arg: i32) -> f32 {
  %res = arith.uitofp %arg : i32 to f32
  return %res : f32
}

func.func @check_uitofp(%arg : i32) -> () {
  %res = func.call @emulate_uitofp(%arg) : (i32) -> (f32)
  vector.print %res : f32
  return
}

func.func @entry() {
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %cst2 = arith.constant 2 : i32
  %cst7 = arith.constant 7 : i32
  %cst1337 = arith.constant 1337 : i32
  %cst_i16_max = arith.constant 65535 : i32
  %cst_i16_overflow = arith.constant 65536 : i32

  %cst_n1 = arith.constant -1 : i32
  %cst_n13 = arith.constant -13 : i32
  %cst_n1337 = arith.constant -1337 : i32

  %cst_i16_min = arith.constant -32768 : i32

  %cst_f32_int_max = arith.constant 16777217 : i32
  %cst_f32_int_min = arith.constant -16777217 : i32

  // CHECK:      0
  func.call @check_uitofp(%cst0) : (i32) -> ()
  // CHECK-NEXT: 1
  func.call @check_uitofp(%cst1) : (i32) -> ()
  // CHECK-NEXT: 2
  func.call @check_uitofp(%cst2) : (i32) -> ()
  // CHECK-NEXT: 7
  func.call @check_uitofp(%cst7) : (i32) -> ()
  // CHECK-NEXT: 1337
  func.call @check_uitofp(%cst1337) : (i32) -> ()
  // CHECK-NEXT: 65535
  func.call @check_uitofp(%cst_i16_max) : (i32) -> ()
  // CHECK-NEXT: 65536
  func.call @check_uitofp(%cst_i16_overflow) : (i32) -> ()

  // CHECK-NEXT: 4.2{{.+}}e+09
  func.call @check_uitofp(%cst_n1) : (i32) -> ()
  // CHECK-NEXT: 4.2{{.+}}e+09
  func.call @check_uitofp(%cst_n1337) : (i32) -> ()

  // CHECK-NEXT: 4.2{{.+}}e+09
  func.call @check_uitofp(%cst_i16_min) : (i32) -> ()
  // CHECK-NEXT: 4.2{{.+}}e+09
  func.call @check_uitofp(%cst_i16_min) : (i32) -> ()
  // CHECK-NEXT: 1.6{{.+}}e+07
  func.call @check_uitofp(%cst_f32_int_max) : (i32) -> ()
  // CHECK-NEXT: 4.2{{.+}}e+09
  func.call @check_uitofp(%cst_f32_int_min) : (i32) -> ()

  return
}

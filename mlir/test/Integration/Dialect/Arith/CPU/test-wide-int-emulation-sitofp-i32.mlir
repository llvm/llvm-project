// Check that the wide integer `arith.sitofp` emulation produces the same result as wide
// `arith.sitofp`. Emulate i32 ops with i16 ops.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// RUN: mlir-opt %s --test-arith-emulate-wide-int="widest-int-supported=16" \
// RUN:             --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// Ops in this function *only* will be emulated using i16 types.
func.func @emulate_sitofp(%arg: i32) -> f32 {
  %res = arith.sitofp %arg : i32 to f32
  return %res : f32
}

func.func @check_sitofp(%arg : i32) -> () {
  %res = func.call @emulate_sitofp(%arg) : (i32) -> (f32)
  vector.print %res : f32
  return
}

func.func @entry() {
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %cst2 = arith.constant 2 : i32
  %cst7 = arith.constant 7 : i32
  %cst1337 = arith.constant 1337 : i32

  %cst_n1 = arith.constant -1 : i32
  %cst_n13 = arith.constant -13 : i32
  %cst_n1337 = arith.constant -1337 : i32

  %cst_i16_min = arith.constant -32768 : i32

  %cst_f32_int_max = arith.constant 16777217 : i32
  %cst_f32_int_min = arith.constant -16777217 : i32

  // CHECK:      0
  func.call @check_sitofp(%cst0) : (i32) -> ()
  // CHECK-NEXT: 1
  func.call @check_sitofp(%cst1) : (i32) -> ()
  // CHECK-NEXT: 2
  func.call @check_sitofp(%cst2) : (i32) -> ()
  // CHECK-NEXT: 7
  func.call @check_sitofp(%cst7) : (i32) -> ()
  // CHECK-NEXT: 1337
  func.call @check_sitofp(%cst1337) : (i32) -> ()
  // CHECK-NEXT: -1
  func.call @check_sitofp(%cst_n1) : (i32) -> ()
  // CHECK-NEXT: -1337
  func.call @check_sitofp(%cst_n1337) : (i32) -> ()

  // CHECK-NEXT: -32768
  func.call @check_sitofp(%cst_i16_min) : (i32) -> ()
  // CHECK-NEXT: 1.6{{.+}}e+07
  func.call @check_sitofp(%cst_f32_int_max) : (i32) -> ()
  // CHECK-NEXT: -1.6{{.+}}e+07
  func.call @check_sitofp(%cst_f32_int_min) : (i32) -> ()

  return
}

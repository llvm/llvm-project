// Ops in this function will be emulated using i16 types.

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

func.func @emulate_subi(%arg: i32, %arg0: i32) -> i32 {
  %res = arith.subi %arg, %arg0 : i32
  return %res : i32
}

func.func @check_subi(%arg : i32, %arg0 : i32) -> () {
  %res = func.call @emulate_subi(%arg, %arg0) : (i32, i32) -> (i32)
  vector.print %res : i32
  return
}

func.func @entry() {
  %lhs1 = arith.constant 1 : i32
  %rhs1 = arith.constant 2 : i32

  // CHECK:       -1
  func.call @check_subi(%lhs1, %rhs1) : (i32, i32) -> ()
  // CHECK-NEXT:  1
  func.call @check_subi(%rhs1, %lhs1) : (i32, i32) -> ()

  %lhs2 = arith.constant 1 : i32
  %rhs2 = arith.constant -2 : i32

  // CHECK-NEXT:  3
  func.call @check_subi(%lhs2, %rhs2) : (i32, i32) -> ()
  // CHECK-NEXT:  -3
  func.call @check_subi(%rhs2, %lhs2) : (i32, i32) -> ()

  %lhs3 = arith.constant -1 : i32
  %rhs3 = arith.constant -2 : i32

  // CHECK-NEXT:  1
  func.call @check_subi(%lhs3, %rhs3) : (i32, i32) -> ()
  // CHECK-NEXT:  -1
  func.call @check_subi(%rhs3, %lhs3) : (i32, i32) -> ()

  // Overflow from the upper/lower part.
  %lhs4 = arith.constant 131074 : i32
  %rhs4 = arith.constant 3 : i32

  // CHECK-NEXT:  131071
  func.call @check_subi(%lhs4, %rhs4) : (i32, i32) -> ()
  // CHECK-NEXT:  -131071
  func.call @check_subi(%rhs4, %lhs4) : (i32, i32) -> ()

  // Overflow in both parts.
  %lhs5 = arith.constant 16385027 : i32
  %rhs5 = arith.constant 16450564 : i32

  // CHECK-NEXT:  -65537
  func.call @check_subi(%lhs5, %rhs5) : (i32, i32) -> ()
  // CHECK-NEXT:  65537
  func.call @check_subi(%rhs5, %lhs5) : (i32, i32) -> ()

  %lhs6 = arith.constant 65536 : i32
  %rhs6 = arith.constant 1 : i32

  // CHECK-NEXT:  65535
  func.call @check_subi(%lhs6, %rhs6) : (i32, i32) -> ()
  // CHECK-NEXT:  -65535
  func.call @check_subi(%rhs6, %lhs6) : (i32, i32) -> ()

  // Max/Min (un)signed integers.
  %sintmax = arith.constant 2147483647 : i32
  %sintmin = arith.constant -2147483648 : i32
  %uintmax = arith.constant -1 : i32
  %uintmin = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32

  // CHECK-NEXT:  -1
  func.call @check_subi(%sintmax, %sintmin) : (i32, i32) -> ()
  // CHECK-NEXT:  1
  func.call @check_subi(%sintmin, %sintmax) : (i32, i32) -> ()
  // CHECK-NEXT:  2147483647
  func.call @check_subi(%sintmin, %cst1) : (i32, i32) -> ()
  // CHECK-NEXT:  -2147483648
  func.call @check_subi(%sintmax, %uintmax) : (i32, i32) -> ()
  // CHECK-NEXT:  -2
  func.call @check_subi(%uintmax, %cst1) : (i32, i32) -> ()
  // CHECK-NEXT:  0
  func.call @check_subi(%uintmax, %uintmax) : (i32, i32) -> ()
  // CHECK-NEXT:  -1
  func.call @check_subi(%uintmin, %cst1) : (i32, i32) -> ()
  // CHECK-NEXT:  1
  func.call @check_subi(%uintmin, %uintmax) : (i32, i32) -> ()

  return
}

// tests arith truncation and extension operations.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

module {
  func.func @extsiOnI1() {
    %true = arith.constant -1 : i1
    %0 = arith.extsi %true : i1 to i16
    vector.print %true : i1
    vector.print %0 : i16
    return
  }

  func.func @extuiOn1I1() {
    %true = arith.constant true
    %0 = arith.extui %true : i1 to i64
    vector.print %true : i1
    vector.print %0 : i64
    return
  }

  func.func @trunciI16ToI8() {
    %c20194_i16 = arith.constant 20194 : i16
    %0 = arith.trunci %c20194_i16 : i16 to i8
    vector.print %c20194_i16 : i16
    vector.print %0 : i8
    return
  }

  func.func @entry() {
    // CHECK:      1
    // CHECK-NEXT: -1
    func.call @extsiOnI1() : () -> ()
    
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    func.call @extuiOn1I1() : () -> ()

    // CHECK-NEXT: 20194
    // CHECK-NEXT: -30
    func.call @trunciI16ToI8() : () -> ()

    return
  }
}
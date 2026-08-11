// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @i1() {
  // printing i1 values
  // print(0 : i1) = '0'; print(1 : i1) = '1'; print(-1 : i1) = '1'
  // CHECK:      0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %false = arith.constant false
  %true = arith.constant 1 : i1
  %true_as_n1 = arith.constant -1 : i1
  vector.print %false : i1
  vector.print %true : i1
  vector.print %true_as_n1 : i1
  return
}

func.func @index() {
  // printing index values
  // print(0 : index) = '0'; print(1 : index) = '1'; print(-1 : index) = '2^w - 1'
  // index constants are printed as unsigned
  // vector.print(arith.constant(x)) ~= toUnsignedRepr x
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 18446744073709551615
  // CHECK-NEXT: 63
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cn1 = arith.constant -1 : index
  %c63 = arith.constant 63 : index

  vector.print %c0 : index
  vector.print %c1 : index
  vector.print %cn1 : index
  vector.print %c63 : index
  return
}

func.func @entry() {
  func.call @i1() : () -> ()
  func.call @index() : () -> ()
  return
}

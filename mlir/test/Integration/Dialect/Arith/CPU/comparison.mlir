// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @cmpi_slt_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@cmpi_slt_i1\n"
  %res = arith.cmpi slt, %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @cmpi_sle_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@cmpi_sle_i1\n"
  %res = arith.cmpi sle, %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @cmpi_sgt_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@cmpi_sgt_i1\n"
  %res = arith.cmpi sgt, %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @cmpi_sge_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@cmpi_sge_i1\n"
  %res = arith.cmpi sge, %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @cmpi_signed() {
  // ------------------------------------------------
  // Test i1
  // ------------------------------------------------
  %false_i1 = arith.constant 0 : i1
  %true_i1 = arith.constant 1 : i1
  %true_i1_n1 = arith.constant -1 : i1

  // sge 0 -1, sge 0 1, should be true
  // since the bitvector `1` is interpreted as the int value -1 in signed comparisons
  // sge 0 -1 == sge 0 1 == true (1)

  // CHECK-LABEL: @cmpi_sge_i1
  // CHECK-NEXT:  1
  func.call @cmpi_sge_i1(%false_i1, %true_i1_n1) : (i1, i1) -> ()

  // CHECK-LABEL: @cmpi_sge_i1
  // CHECK-NEXT:  1
  func.call @cmpi_sge_i1(%false_i1, %true_i1) : (i1, i1) -> ()

  %false = arith.constant false
  %true = arith.constant true

  // slt 0 1 = false, sle 0 1 = false, sgt 0 1 = true, sge 0 1 = true, sge 1 0 = false

  // CHECK-LABEL: @cmpi_slt_i1
  // CHECK-NEXT:  0
  func.call @cmpi_slt_i1(%false, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @cmpi_sle_i1
  // CHECK-NEXT:  0
  func.call @cmpi_sle_i1(%false, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @cmpi_sgt_i1
  // CHECK-NEXT:  1
  func.call @cmpi_sgt_i1(%false, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @cmpi_sge_i1
  // CHECK-NEXT:  1
  func.call @cmpi_sge_i1(%false, %true) : (i1, i1) -> ()
  
  // CHECK-LABEL: @cmpi_sge_i1
  // CHECK-NEXT:  0
  func.call @cmpi_sge_i1(%true, %false) : (i1, i1) -> ()
  
  // ------------------------------------------------
  // TODO: Test i8, i16 etc..
  // ------------------------------------------------
  return
}

func.func @cmpi_ult_index(%v1 : index, %v2 : index) {
  vector.print str "@cmpi_ult_index\n"
  %res = arith.cmpi ult, %v1, %v2 : index
  vector.print %res : i1
  return
}

func.func @cmpi_unsigned() {
  // ------------------------------------------------
  // Test index
  // ------------------------------------------------
  // 0 `ult` -2^63 = true
  %zero = arith.constant 0 : index
  %index_min = arith.constant -9223372036854775808 : index

  // CHECK-LABEL: @cmpi_ult_index
  // CHECK-NEXT: 1
  func.call @cmpi_ult_index(%zero, %index_min) : (index, index) -> ()
  
  // ------------------------------------------------
  // TODO: i1, i8, i16, uge, ule etc.. 
  // ------------------------------------------------
  return
}

func.func @entry() {
  func.call @cmpi_signed() : () -> ()
  func.call @cmpi_unsigned() : () -> ()
  return
}

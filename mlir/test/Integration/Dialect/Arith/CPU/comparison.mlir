// Tests comparison operations.
// These tests are intended to be target agnostic: they should yield the same results 
// regardless of the target platform.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @slt_cmpi_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@slt_cmpi_i1\n"
  %res = arith.cmpi slt, %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @sle_cmpi_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@sle_cmpi_i1\n"
  %res = arith.cmpi sle, %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @sgt_cmpi_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@sgt_cmpi_i1\n"
  %res = arith.cmpi sgt, %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @sge_cmpi_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@sge_cmpi_i1\n"
  %res = arith.cmpi sge, %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @signed_cmpi() {
  // ------------------------------------------------
  // Test i1
  // ------------------------------------------------
  %false_i1 = arith.constant 0 : i1
  %true_i1 = arith.constant 1 : i1
  %true_i1_n1 = arith.constant -1 : i1

  // sge 0 -1, sge 0 1, should be true
  // sge 0 -1 == sge 0 1 == true (1)

  // CHECK-LABEL: @sge_cmpi_i1
  // CHECK-NEXT:  1
  func.call @sge_cmpi_i1(%false_i1, %true_i1_n1) : (i1, i1) -> ()

  // CHECK-LABEL: @sge_cmpi_i1
  // CHECK-NEXT:  1
  func.call @sge_cmpi_i1(%false_i1, %true_i1) : (i1, i1) -> ()

  %false = arith.constant false
  %true = arith.constant true

  // signed comparisons on i1s
  // slt 0 1 = false, sle 0 1 = false, sgt 0 1 = true, sge 0 1 = true

  // CHECK-LABEL: @slt_cmpi_i1
  // CHECK-NEXT:  0
  func.call @slt_cmpi_i1(%false, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @sle_cmpi_i1
  // CHECK-NEXT:  0
  func.call @sle_cmpi_i1(%false, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @sgt_cmpi_i1
  // CHECK-NEXT:  1
  func.call @sgt_cmpi_i1(%false, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @sge_cmpi_i1
  // CHECK-NEXT:  1
  func.call @sge_cmpi_i1(%false, %true) : (i1, i1) -> ()

  // check that addui_extended overflow bit is treated as -1 in comparison operations
  //  in the case of an overflow
  // addui_extended -1 -1 = (..., overflow_bit) 
  // assert(overflow_bit <= 0)
  %n1_i64 = arith.constant -1 : i64
  %sum, %overflow = arith.addui_extended %n1_i64, %n1_i64 : i64, i1

  // CHECK-LABEL: @sge_cmpi_i1
  // CHECK-NEXT: 0
  func.call @sge_cmpi_i1(%overflow, %false) : (i1, i1) -> ()
  
  // ------------------------------------------------
  // Test i8, i16 etc.. TODO
  // ------------------------------------------------
  return
}

func.func @ult_cmpi_index(%v1 : index, %v2 : index) {
  vector.print str "@ult_cmpi_index\n"
  %res = arith.cmpi ult, %v1, %v2 : index
  vector.print %res : i1
  return
}

func.func @unsigned_cmpi() {
  // ------------------------------------------------
  // Test index
  // ------------------------------------------------
  // 0 `ult` -2^63 = true
  %zero = arith.constant 0 : index
  %index_min = arith.constant -9223372036854775808 : index

  // CHECK-LABEL: @ult_cmpi_index
  // CHECK-NEXT: 1
  func.call @ult_cmpi_index(%zero, %index_min) : (index, index) -> ()
  
  // ------------------------------------------------
  // Test i1, i8, i16 etc.. TODO
  // ------------------------------------------------
  return
}

func.func @entry() {
  func.call @signed_cmpi() : () -> ()
  func.call @unsigned_cmpi() : () -> ()
  return
}

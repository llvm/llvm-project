// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Test OpAsmOpInterface
//===----------------------------------------------------------------------===//

func.func @result_name_from_op_asm_type_interface() {
  // CHECK-LABEL: @result_name_from_op_asm_type_interface
  // CHECK: %op_asm_type_interface
  %0 = "test.result_name_from_type"() : () -> !test.op_asm_type_interface
  return
}

// -----

func.func @block_argument_name_from_op_asm_type_interface() {
  // CHECK-LABEL: @block_argument_name_from_op_asm_type_interface
  // CHECK: ^bb0(%op_asm_type_interface
  test.block_argument_name_from_type {
    ^bb0(%arg0: !test.op_asm_type_interface):
      "test.terminator"() : ()->()
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test OpAsmTypeInterface
//===----------------------------------------------------------------------===//

func.func @result_name_from_op_asm_type_interface_asmprinter() {
  // CHECK-LABEL: @result_name_from_op_asm_type_interface_asmprinter
  // CHECK: %op_asm_type_interface
  %0 = "test.result_name_from_type_interface"() : () -> !test.op_asm_type_interface
  return
}

// -----

// i1 does not have OpAsmTypeInterface, should not get named.
func.func @result_name_from_op_asm_type_interface_not_all() {
  // CHECK-LABEL: @result_name_from_op_asm_type_interface_not_all
  // CHECK-NOT: %op_asm_type_interface
  // CHECK: %0:2
  %0:2 = "test.result_name_from_type_interface"() : () -> (!test.op_asm_type_interface, i1)
  return
}

// -----

func.func @block_argument_name_from_op_asm_type_interface_asmprinter() {
  // CHECK-LABEL: @block_argument_name_from_op_asm_type_interface_asmprinter
  // CHECK: ^bb0(%op_asm_type_interface
  test.block_argument_name_from_type_interface {
    ^bb0(%arg0: !test.op_asm_type_interface):
      "test.terminator"() : ()->()
  }
  return
}
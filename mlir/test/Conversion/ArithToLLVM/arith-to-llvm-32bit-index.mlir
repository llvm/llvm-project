// RUN: mlir-opt %s --convert-arith-to-llvm -split-input-file | FileCheck %s

// Verify that ArithToLLVMConversionPass respects the module's data layout
// when deriving the index type.  When the module declares a 32-bit index via
// dlti.dl_spec, index constants and index arithmetic ops must be emitted as
// i32 instead of the default i64.

// -----

// 32-bit data layout: arith.constant with index type -> i32 constant.

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {

func.func @constant_index_32bit() -> index {
  %c0 = arith.constant 0 : index
  return %c0 : index
}

}

// CHECK-LABEL: func @constant_index_32bit
// CHECK: llvm.mlir.constant(0 : index) : i32

// -----

// 32-bit data layout: arith.cmpi on index type -> icmp on i32.

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {

func.func @cmpi_index_32bit(%a: index, %b: index) -> i1 {
  %cmp = arith.cmpi slt, %a, %b : index
  return %cmp : i1
}

}

// CHECK-LABEL: func @cmpi_index_32bit
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i32
// CHECK: llvm.icmp "slt" %{{.*}}, %{{.*}} : i32

// -----

// 32-bit data layout: arith.addi on index type -> add on i32.

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {

func.func @addi_index_32bit(%a: index, %b: index) -> index {
  %add = arith.addi %a, %b : index
  return %add : index
}

}

// CHECK-LABEL: func @addi_index_32bit
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i32
// CHECK: llvm.add %{{.*}}, %{{.*}} : i32

// -----

// Without dlti.dl_spec the default index width (i64) is preserved.

module {

func.func @constant_index_default() -> index {
  %c0 = arith.constant 0 : index
  return %c0 : index
}

}

// CHECK-LABEL: func @constant_index_default
// CHECK: llvm.mlir.constant(0 : index) : i64

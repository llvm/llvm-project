// RUN: mlir-opt -emit-bytecode %s | mlir-opt --mlir-print-debuginfo | FileCheck %s

// CHECK: llvm.di_subprogram
// CHECK-NOT: llvm.di_subprogram
#di_file = #llvm.di_file<"foo.c" in "/mlir/">
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "a", file = #di_file, line = 2, type = #di_basic_type>

module attributes {test.alias = #di_local_variable} {
} loc(fused<#di_subprogram>[])
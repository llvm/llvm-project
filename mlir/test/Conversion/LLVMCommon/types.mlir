// RUN: mlir-opt %s --convert-func-to-llvm --split-input-file | FileCheck %s

// CHECK: @create_clashes_on_conversion(!llvm.struct<"foo", (index)>)
func.func private @clashing_struct_name(!llvm.struct<"_Converted.foo", (f32)>)
func.func private @create_clashes_on_conversion(!llvm.struct<"foo", (index)>)

// -----

// CHECK: @merge_on_conversion(!llvm.struct<"_Converted.foo", (i64)>) attributes {sym_visibility = "private"}
func.func private @clashing_struct_name(!llvm.struct<"_Converted.foo", (i64)>)
func.func private @merge_on_conversion(!llvm.struct<"foo", (index)>)

// -----

// CHECK: @create_clashes_on_conversion_recursive(!llvm.struct<"foo", (!llvm.struct<"foo">, index)>)
func.func private @clashing_struct_name(!llvm.struct<"_Converted.foo", (struct<"_Converted.foo">, f32)>)
func.func private @create_clashes_on_conversion_recursive(!llvm.struct<"foo", (struct<"foo">, index)>)

// -----

// CHECK: @merge_on_conversion_recursive(!llvm.struct<"_Converted.foo", (struct<"_Converted.foo">, i64)>)
func.func private @clashing_struct_name(!llvm.struct<"_Converted.foo", (struct<"_Converted.foo">, i64)>)
func.func private @merge_on_conversion_recursive(!llvm.struct<"foo", (struct<"foo">, index)>)

// -----

// CHECK: @create_clashing_pack(!llvm.struct<"foo", packed (!llvm.struct<"foo">, index)>)
func.func private @clashing_struct_name(!llvm.struct<"_Converted.foo", (struct<"_Converted.foo">, i64)>)
func.func private @create_clashing_pack(!llvm.struct<"foo", packed (struct<"foo">, index)>)

// -----

// CHECK: @merge_on_conversion_pack(!llvm.struct<"_Converted.foo", packed (struct<"_Converted.foo">, i64)>)
func.func private @clashing_struct_name(!llvm.struct<"_Converted.foo", packed (struct<"_Converted.foo">, i64)>)
func.func private @merge_on_conversion_pack(!llvm.struct<"foo", packed (struct<"foo">, index)>)

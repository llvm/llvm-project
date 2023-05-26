// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
    spirv.func @linkage_attr_test_kernel()  "DontInline"  attributes {}  {
        %uchar_0 = spirv.Constant 0 : i8
        %ushort_1 = spirv.Constant 1 : i16
        %uint_0 = spirv.Constant 0 : i32
        spirv.FunctionCall @outside.func.with.linkage(%uchar_0):(i8) -> ()
        spirv.Return
    }
    // CHECK: linkage_attributes = #spirv.linkage_attributes<linkage_name = outside.func, linkage_type = <Import>>
    spirv.func @outside.func.with.linkage(%arg0 : i8) -> () "Pure" attributes {
      linkage_attributes=#spirv.linkage_attributes<
        linkage_name="outside.func",
        linkage_type=<Import>
      >
    }
    spirv.func @inside.func() -> () "Pure" attributes {} {spirv.Return}
}

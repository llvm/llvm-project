// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

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

// -----

// CHECK: spirv.func @arg_decoration_pointer(%{{.+}}: !spirv.ptr<i32, PhysicalStorageBuffer> {spirv.decoration = #spirv.decoration<Aliased>}) "None"
spirv.func @arg_decoration_pointer(%arg0: !spirv.ptr<i32, PhysicalStorageBuffer> { spirv.decoration = #spirv.decoration<Aliased> }) "None" {
  spirv.Return
}

// -----

// CHECK: spirv.func @arg_decoration_pointer(%{{.+}}: !spirv.ptr<i32, PhysicalStorageBuffer> {spirv.decoration = #spirv.decoration<Restrict>}) "None"
spirv.func @arg_decoration_pointer(%arg0: !spirv.ptr<i32, PhysicalStorageBuffer> { spirv.decoration = #spirv.decoration<Restrict> }) "None" {
  spirv.Return
}

// -----

// CHECK: spirv.func @arg_decoration_pointer(%{{.+}}: !spirv.ptr<!spirv.ptr<i32, PhysicalStorageBuffer>, Generic> {spirv.decoration = #spirv.decoration<AliasedPointer>}) "None"
spirv.func @arg_decoration_pointer(%arg0: !spirv.ptr<!spirv.ptr<i32, PhysicalStorageBuffer>, Generic> { spirv.decoration = #spirv.decoration<AliasedPointer> }) "None" {
  spirv.Return
}

// -----

// CHECK: spirv.func @arg_decoration_pointer(%{{.+}}: !spirv.ptr<!spirv.ptr<i32, PhysicalStorageBuffer>, Generic> {spirv.decoration = #spirv.decoration<RestrictPointer>}) "None"
spirv.func @arg_decoration_pointer(%arg0: !spirv.ptr<!spirv.ptr<i32, PhysicalStorageBuffer>, Generic> { spirv.decoration = #spirv.decoration<RestrictPointer> }) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.func' op with physical buffer pointer must be decorated either 'Aliased' or 'Restrict'}}
spirv.func @no_arg_decoration_pointer(%arg0: !spirv.ptr<i32, PhysicalStorageBuffer>) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.func' op with a pointer points to a physical buffer pointer must be decorated either 'AliasedPointer' or 'RestrictPointer'}}
spirv.func @no_arg_decoration_pointer(%arg0: !spirv.ptr<!spirv.ptr<i32, PhysicalStorageBuffer>, Function>) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.func' op with physical buffer pointer must be decorated either 'Aliased' or 'Restrict'}}
spirv.func @no_decoration_name_attr(%arg0 : !spirv.ptr<i32, PhysicalStorageBuffer> { random_attr = #spirv.decoration<Aliased> }) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.func' op arguments may only have dialect attributes}}
spirv.func @no_decoration_name_attr(%arg0 : !spirv.ptr<i32, PhysicalStorageBuffer> { spirv.decoration = #spirv.decoration<Restrict>, random_attr = #spirv.decoration<Aliased> }) "None" {
  spirv.Return
}

// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file -verify-diagnostics %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
    spirv.func @linkage_attr_test_kernel()  "DontInline"  attributes {}  {
        %uchar_0 = spirv.Constant 0 : i8
        %ushort_1 = spirv.Constant 1 : i16
        %uint_0 = spirv.Constant 0 : i32
        spirv.FunctionCall @outside.func.with.linkage(%uchar_0):(i8) -> ()
        spirv.Return
    }
    // CHECK: linkage_attributes = #spirv.linkage_attributes<linkage_name = "outside.func", linkage_type = <Import>>
    spirv.func @outside.func.with.linkage(%arg0 : i8) -> () "Pure" attributes {
      linkage_attributes=#spirv.linkage_attributes<
        linkage_name="outside.func",
        linkage_type=<Import>
      >
    }
    spirv.func @inside.func() -> () "Pure" attributes {} {spirv.Return}
}

// -----

spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.0,
    [Shader, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]> {
  // CHECK-LABEL: spirv.func @func_arg_decoration_aliased(%{{.*}}: !spirv.ptr<i32, PhysicalStorageBuffer> {spirv.decoration = #spirv.decoration<Aliased>})
  spirv.func @func_arg_decoration_aliased(
      %arg0 : !spirv.ptr<i32, PhysicalStorageBuffer> { spirv.decoration = #spirv.decoration<Aliased> }
  ) "None" {
    spirv.Return
  }
}

// -----

spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.0,
    [Shader, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]> {
  // CHECK-LABEL: spirv.func @func_arg_decoration_restrict(%{{.*}}: !spirv.ptr<i32, PhysicalStorageBuffer> {spirv.decoration = #spirv.decoration<Restrict>})
  spirv.func @func_arg_decoration_restrict(
      %arg0 : !spirv.ptr<i32,PhysicalStorageBuffer> { spirv.decoration = #spirv.decoration<Restrict> }
  ) "None" {
    spirv.Return
  }
}

// -----

spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.0,
    [Shader, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]> {
  // CHECK-LABEL: spirv.func @func_arg_decoration_aliased_pointer(%{{.*}}: !spirv.ptr<!spirv.ptr<i32, PhysicalStorageBuffer>, Generic> {spirv.decoration = #spirv.decoration<AliasedPointer>})
  spirv.func @func_arg_decoration_aliased_pointer(
      %arg0 : !spirv.ptr<!spirv.ptr<i32,PhysicalStorageBuffer>, Generic> { spirv.decoration = #spirv.decoration<AliasedPointer> }
  ) "None" {
    spirv.Return
  }
}

// -----

spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.0,
    [Shader, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]> {
  // CHECK-LABEL: spirv.func @func_arg_decoration_restrict_pointer(%{{.*}}: !spirv.ptr<!spirv.ptr<i32, PhysicalStorageBuffer>, Generic> {spirv.decoration = #spirv.decoration<RestrictPointer>})
  spirv.func @func_arg_decoration_restrict_pointer(
      %arg0 : !spirv.ptr<!spirv.ptr<i32,PhysicalStorageBuffer>, Generic> { spirv.decoration = #spirv.decoration<RestrictPointer> }
  ) "None" {
    spirv.Return
  }
}

// -----

spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.0,
    [Shader, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]> {
  // CHECK-LABEL: spirv.func @fn1(%{{.*}}: i32, %{{.*}}: !spirv.ptr<i32, PhysicalStorageBuffer> {spirv.decoration = #spirv.decoration<Aliased>})
  spirv.func @fn1(
      %arg0: i32,
      %arg1: !spirv.ptr<i32, PhysicalStorageBuffer> { spirv.decoration = #spirv.decoration<Aliased> }
  ) "None" {
    spirv.Return
  }

  // CHECK-LABEL: spirv.func @fn2(%{{.*}}: !spirv.ptr<i32, PhysicalStorageBuffer> {spirv.decoration = #spirv.decoration<Aliased>}, %{{.*}}: !spirv.ptr<i32, PhysicalStorageBuffer> {spirv.decoration = #spirv.decoration<Restrict>})
  spirv.func @fn2(
      %arg0: !spirv.ptr<i32, PhysicalStorageBuffer> { spirv.decoration = #spirv.decoration<Aliased> },
      %arg1: !spirv.ptr<i32, PhysicalStorageBuffer> { spirv.decoration = #spirv.decoration<Restrict>}
  ) "None" {
    spirv.Return
  }
}

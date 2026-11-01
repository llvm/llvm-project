// RUN: mlir-opt %s -convert-spirv-to-llvm -verify-diagnostics -split-input-file

//===----------------------------------------------------------------------===//
// spirv.CopyMemory
//===----------------------------------------------------------------------===//

// A runtime array has no static size, so there is no byte count for memcpy.
spirv.func @copy_memory_rtarray(%arg0: !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>, %arg1: !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.CopyMemory' that was explicitly marked illegal}}
  spirv.CopyMemory "StorageBuffer" %arg0, "StorageBuffer" %arg1 : !spirv.rtarray<f32>
  spirv.Return
}

// -----

// A runtime array nested in a struct makes the whole struct unsized.
spirv.func @copy_memory_struct_with_rtarray(%arg0: !spirv.ptr<!spirv.struct<(i32, !spirv.rtarray<f32>)>, StorageBuffer>, %arg1: !spirv.ptr<!spirv.struct<(i32, !spirv.rtarray<f32>)>, StorageBuffer>) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.CopyMemory' that was explicitly marked illegal}}
  spirv.CopyMemory "StorageBuffer" %arg0, "StorageBuffer" %arg1 : !spirv.struct<(i32, !spirv.rtarray<f32>)>
  spirv.Return
}

// -----

// Memory-model flags have no llvm.intr.memcpy counterpart.
spirv.func @copy_memory_non_private_pointer(%arg0: !spirv.ptr<f32, StorageBuffer>, %arg1: !spirv.ptr<f32, StorageBuffer>) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.CopyMemory' that was explicitly marked illegal}}
  spirv.CopyMemory "StorageBuffer" %arg0, "StorageBuffer" %arg1 ["NonPrivatePointer"] : f32
  spirv.Return
}

// -----

// The Source mask is checked too, not only the Target mask.
spirv.func @copy_memory_unsupported_source_mask(%arg0: !spirv.ptr<f32, StorageBuffer>, %arg1: !spirv.ptr<f32, StorageBuffer>) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.CopyMemory' that was explicitly marked illegal}}
  spirv.CopyMemory "StorageBuffer" %arg0, "StorageBuffer" %arg1 ["Aligned", 4], ["NonPrivatePointer|MakePointerVisible"] : f32
  spirv.Return
}

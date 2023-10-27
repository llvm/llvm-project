// RUN: mlir-opt %s -convert-spirv-to-llvm='use-opaque-pointers=1' -verify-diagnostics -split-input-file

// expected-error@+1 {{failed to legalize operation 'spirv.func' that was explicitly marked illegal}}
spirv.func @array_with_unnatural_stride(%arg: !spirv.array<4 x f32, stride=8>) -> () "None" {
  spirv.Return
}

// -----

// expected-error@+1 {{failed to legalize operation 'spirv.func' that was explicitly marked illegal}}
spirv.func @struct_with_unnatural_offset(%arg: !spirv.struct<(i32[0], i32[8])>) -> () "None" {
  spirv.Return
}

// -----

// expected-error@+1 {{failed to legalize operation 'spirv.func' that was explicitly marked illegal}}
spirv.func @struct_with_decorations(%arg: !spirv.struct<(f32 [RelaxedPrecision])>) -> () "None" {
  spirv.Return
}

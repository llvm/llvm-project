// RUN: mlir-opt %s -convert-spirv-to-llvm -verify-diagnostics -split-input-file

// expected-error@+1 {{failed to legalize operation 'spv.func' that was explicitly marked illegal}}
spv.func @array_with_stride(%arg: !spv.array<4 x f32, stride=4>) -> () "None" {
  spv.Return
}

// -----

// expected-error@+1 {{failed to legalize operation 'spv.func' that was explicitly marked illegal}}
spv.func @struct_with_offset1(%arg: !spv.struct<i32[0], i32[4]>) -> () "None" {
  spv.Return
}

// -----

// expected-error@+1 {{failed to legalize operation 'spv.func' that was explicitly marked illegal}}
spv.func @struct_with_offset2(%arg: !spv.struct<i32[0], i32[8]>) -> () "None" {
  spv.Return
}

// -----

// expected-error@+1 {{failed to legalize operation 'spv.func' that was explicitly marked illegal}}
spv.func @struct_with_decorations(%arg: !spv.struct<f32 [RelaxedPrecision]>) -> () "None" {
  spv.Return
}

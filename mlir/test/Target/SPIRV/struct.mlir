// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Addresses, Float64, Int64, Linkage], [SPV_KHR_storage_buffer_storage_class]> {
  // CHECK: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Input>
  spirv.GlobalVariable @var0 bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Input>

  // CHECK: !spirv.ptr<!spirv.struct<(f32 [0], !spirv.struct<(f32 [0], !spirv.array<16 x f32, stride=4> [4])> [4])>, Input>
  spirv.GlobalVariable @var1 bind(0, 2) : !spirv.ptr<!spirv.struct<(f32 [0], !spirv.struct<(f32 [0], !spirv.array<16 x f32, stride=4> [4])> [4])>, Input>

  // CHECK: !spirv.ptr<!spirv.struct<(f32 [0], i32 [4], f64 [8], i64 [16], f32 [24], i32 [30], f32 [34], i32 [38]), Block>, StorageBuffer>
  spirv.GlobalVariable @var2 : !spirv.ptr<!spirv.struct<(f32 [0], i32 [4], f64 [8], i64 [16], f32 [24], i32 [30], f32 [34], i32 [38]), Block>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<(!spirv.array<128 x !spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, stride=512> [0]), Block>, StorageBuffer>
  spirv.GlobalVariable @var3 : !spirv.ptr<!spirv.struct<(!spirv.array<128 x !spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, stride=512> [0]), Block>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<(f32 [0, NonWritable], i32 [4]), Block>, StorageBuffer>
  spirv.GlobalVariable @var4 : !spirv.ptr<!spirv.struct<(f32 [0, NonWritable], i32 [4]), Block>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<(f32 [0, NonWritable], i32 [4, NonWritable, NonReadable]), Block>, StorageBuffer>
  spirv.GlobalVariable @var5 : !spirv.ptr<!spirv.struct<(f32 [0, NonWritable], i32 [4, NonWritable, NonReadable]), Block>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<(f32 [0, NonWritable], i32 [4, NonWritable, NonReadable]), Block>, StorageBuffer>
  spirv.GlobalVariable @var6 : !spirv.ptr<!spirv.struct<(f32 [0, NonWritable], i32 [4, NonWritable, NonReadable]), Block>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16]), Block>, StorageBuffer>
  spirv.GlobalVariable @var7 : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16]), Block>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<()>, StorageBuffer>
  spirv.GlobalVariable @empty : !spirv.ptr<!spirv.struct<()>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<empty_struct, ()>, StorageBuffer>
  spirv.GlobalVariable @id_empty : !spirv.ptr<!spirv.struct<empty_struct, ()>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<test_id, (!spirv.array<128 x f32, stride=4> [0])>, Input>
  spirv.GlobalVariable @id_var0 : !spirv.ptr<!spirv.struct<test_id, (!spirv.array<128 x f32, stride=4> [0])>, Input>

  // CHECK: !spirv.ptr<!spirv.struct<rec, (!spirv.ptr<!spirv.struct<rec>, StorageBuffer> [0]), Block>, StorageBuffer>
  spirv.GlobalVariable @recursive_simple : !spirv.ptr<!spirv.struct<rec, (!spirv.ptr<!spirv.struct<rec>, StorageBuffer> [0]), Block>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<a, (!spirv.ptr<!spirv.struct<b, (!spirv.ptr<!spirv.struct<a>, Uniform> [0]), Block>, Uniform> [0]), Block>, Uniform>
  spirv.GlobalVariable @recursive_2 : !spirv.ptr<!spirv.struct<a, (!spirv.ptr<!spirv.struct<b, (!spirv.ptr<!spirv.struct<a>, Uniform> [0]), Block>, Uniform> [0]), Block>, Uniform>

  // CHECK: !spirv.ptr<!spirv.struct<axx, (!spirv.ptr<!spirv.struct<bxx, (!spirv.ptr<!spirv.struct<axx>, Uniform> [0], !spirv.ptr<!spirv.struct<bxx>, Uniform> [8]), Block>, Uniform> [0]), Block>, Uniform>
  spirv.GlobalVariable @recursive_3 : !spirv.ptr<!spirv.struct<axx, (!spirv.ptr<!spirv.struct<bxx, (!spirv.ptr<!spirv.struct<axx>, Uniform> [0], !spirv.ptr<!spirv.struct<bxx>, Uniform> [8]), Block>, Uniform> [0]), Block>, Uniform>

  // CHECK: spirv.GlobalVariable @block : !spirv.ptr<!spirv.struct<vert, (vector<4xf32> [BuiltIn=0], f32 [BuiltIn=1]), Block>, Output>
  spirv.GlobalVariable @block : !spirv.ptr<!spirv.struct<vert, (vector<4xf32> [BuiltIn=0], f32 [BuiltIn=1]), Block>, Output>

  // CHECK: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Input>,
  // CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Output>
  spirv.func @kernel(%arg0: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Input>, %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Output>) -> () "None" {
    spirv.Return
  }
}

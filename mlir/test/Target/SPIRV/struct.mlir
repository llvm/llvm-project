// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Input>
  spirv.GlobalVariable @var0 bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Input>

  // CHECK: !spirv.ptr<!spirv.struct<(f32 [0], !spirv.struct<(f32 [0], !spirv.array<16 x f32, stride=4> [4])> [4])>, Input>
  spirv.GlobalVariable @var1 bind(0, 2) : !spirv.ptr<!spirv.struct<(f32 [0], !spirv.struct<(f32 [0], !spirv.array<16 x f32, stride=4> [4])> [4])>, Input>

  // CHECK: !spirv.ptr<!spirv.struct<(f32 [0], i32 [4], f64 [8], i64 [16], f32 [24], i32 [30], f32 [34], i32 [38])>, StorageBuffer>
  spirv.GlobalVariable @var2 : !spirv.ptr<!spirv.struct<(f32 [0], i32 [4], f64 [8], i64 [16], f32 [24], i32 [30], f32 [34], i32 [38])>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<(!spirv.array<128 x !spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, stride=512> [0])>, StorageBuffer>
  spirv.GlobalVariable @var3 : !spirv.ptr<!spirv.struct<(!spirv.array<128 x !spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, stride=512> [0])>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<(f32 [0, NonWritable], i32 [4])>, StorageBuffer>
  spirv.GlobalVariable @var4 : !spirv.ptr<!spirv.struct<(f32 [0, NonWritable], i32 [4])>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<(f32 [NonWritable], i32 [NonWritable, NonReadable])>, StorageBuffer>
  spirv.GlobalVariable @var5 : !spirv.ptr<!spirv.struct<(f32 [NonWritable], i32 [NonWritable, NonReadable])>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<(f32 [0, NonWritable], i32 [4, NonWritable, NonReadable])>, StorageBuffer>
  spirv.GlobalVariable @var6 : !spirv.ptr<!spirv.struct<(f32 [0, NonWritable], i32 [4, NonWritable, NonReadable])>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16])>, StorageBuffer>
  spirv.GlobalVariable @var7 : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16])>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<()>, StorageBuffer>
  spirv.GlobalVariable @empty : !spirv.ptr<!spirv.struct<()>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<empty_struct, ()>, StorageBuffer>
  spirv.GlobalVariable @id_empty : !spirv.ptr<!spirv.struct<empty_struct, ()>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<test_id, (!spirv.array<128 x f32, stride=4> [0])>, Input>
  spirv.GlobalVariable @id_var0 : !spirv.ptr<!spirv.struct<test_id, (!spirv.array<128 x f32, stride=4> [0])>, Input>


  // CHECK: !spirv.ptr<!spirv.struct<rec, (!spirv.ptr<!spirv.struct<rec>, StorageBuffer>)>, StorageBuffer>
  spirv.GlobalVariable @recursive_simple : !spirv.ptr<!spirv.struct<rec, (!spirv.ptr<!spirv.struct<rec>, StorageBuffer>)>, StorageBuffer>

  // CHECK: !spirv.ptr<!spirv.struct<a, (!spirv.ptr<!spirv.struct<b, (!spirv.ptr<!spirv.struct<a>, Uniform>)>, Uniform>)>, Uniform>
  spirv.GlobalVariable @recursive_2 : !spirv.ptr<!spirv.struct<a, (!spirv.ptr<!spirv.struct<b, (!spirv.ptr<!spirv.struct<a>, Uniform>)>, Uniform>)>, Uniform>

  // CHECK: !spirv.ptr<!spirv.struct<axx, (!spirv.ptr<!spirv.struct<bxx, (!spirv.ptr<!spirv.struct<axx>, Uniform>, !spirv.ptr<!spirv.struct<bxx>, Uniform>)>, Uniform>)>, Uniform>
  spirv.GlobalVariable @recursive_3 : !spirv.ptr<!spirv.struct<axx, (!spirv.ptr<!spirv.struct<bxx, (!spirv.ptr<!spirv.struct<axx>, Uniform>, !spirv.ptr<!spirv.struct<bxx>, Uniform>)>, Uniform>)>, Uniform>

  // CHECK: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Input>,
  // CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Output>
  spirv.func @kernel(%arg0: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Input>, %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, Output>) -> () "None" {
    spirv.Return
  }
}

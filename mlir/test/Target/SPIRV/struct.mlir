// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: !spv.ptr<!spv.struct<(!spv.array<128 x f32, stride=4> [0])>, Input>
  spv.GlobalVariable @var0 bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<128 x f32, stride=4> [0])>, Input>

  // CHECK: !spv.ptr<!spv.struct<(f32 [0], !spv.struct<(f32 [0], !spv.array<16 x f32, stride=4> [4])> [4])>, Input>
  spv.GlobalVariable @var1 bind(0, 2) : !spv.ptr<!spv.struct<(f32 [0], !spv.struct<(f32 [0], !spv.array<16 x f32, stride=4> [4])> [4])>, Input>

  // CHECK: !spv.ptr<!spv.struct<(f32 [0], i32 [4], f64 [8], i64 [16], f32 [24], i32 [30], f32 [34], i32 [38])>, StorageBuffer>
  spv.GlobalVariable @var2 : !spv.ptr<!spv.struct<(f32 [0], i32 [4], f64 [8], i64 [16], f32 [24], i32 [30], f32 [34], i32 [38])>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<(!spv.array<128 x !spv.struct<(!spv.array<128 x f32, stride=4> [0])>, stride=512> [0])>, StorageBuffer>
  spv.GlobalVariable @var3 : !spv.ptr<!spv.struct<(!spv.array<128 x !spv.struct<(!spv.array<128 x f32, stride=4> [0])>, stride=512> [0])>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<(f32 [0, NonWritable], i32 [4])>, StorageBuffer>
  spv.GlobalVariable @var4 : !spv.ptr<!spv.struct<(f32 [0, NonWritable], i32 [4])>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<(f32 [NonWritable], i32 [NonWritable, NonReadable])>, StorageBuffer>
  spv.GlobalVariable @var5 : !spv.ptr<!spv.struct<(f32 [NonWritable], i32 [NonWritable, NonReadable])>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<(f32 [0, NonWritable], i32 [4, NonWritable, NonReadable])>, StorageBuffer>
  spv.GlobalVariable @var6 : !spv.ptr<!spv.struct<(f32 [0, NonWritable], i32 [4, NonWritable, NonReadable])>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<(!spv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16])>, StorageBuffer>
  spv.GlobalVariable @var7 : !spv.ptr<!spv.struct<(!spv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16])>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<()>, StorageBuffer>
  spv.GlobalVariable @empty : !spv.ptr<!spv.struct<()>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<empty_struct, ()>, StorageBuffer>
  spv.GlobalVariable @id_empty : !spv.ptr<!spv.struct<empty_struct, ()>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<test_id, (!spv.array<128 x f32, stride=4> [0])>, Input>
  spv.GlobalVariable @id_var0 : !spv.ptr<!spv.struct<test_id, (!spv.array<128 x f32, stride=4> [0])>, Input>


  // CHECK: !spv.ptr<!spv.struct<rec, (!spv.ptr<!spv.struct<rec>, StorageBuffer>)>, StorageBuffer>
  spv.GlobalVariable @recursive_simple : !spv.ptr<!spv.struct<rec, (!spv.ptr<!spv.struct<rec>, StorageBuffer>)>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<a, (!spv.ptr<!spv.struct<b, (!spv.ptr<!spv.struct<a>, Uniform>)>, Uniform>)>, Uniform>
  spv.GlobalVariable @recursive_2 : !spv.ptr<!spv.struct<a, (!spv.ptr<!spv.struct<b, (!spv.ptr<!spv.struct<a>, Uniform>)>, Uniform>)>, Uniform>

  // CHECK: !spv.ptr<!spv.struct<axx, (!spv.ptr<!spv.struct<bxx, (!spv.ptr<!spv.struct<axx>, Uniform>, !spv.ptr<!spv.struct<bxx>, Uniform>)>, Uniform>)>, Uniform>
  spv.GlobalVariable @recursive_3 : !spv.ptr<!spv.struct<axx, (!spv.ptr<!spv.struct<bxx, (!spv.ptr<!spv.struct<axx>, Uniform>, !spv.ptr<!spv.struct<bxx>, Uniform>)>, Uniform>)>, Uniform>

  // CHECK: !spv.ptr<!spv.struct<(!spv.array<128 x f32, stride=4> [0])>, Input>,
  // CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<128 x f32, stride=4> [0])>, Output>
  spv.func @kernel(%arg0: !spv.ptr<!spv.struct<(!spv.array<128 x f32, stride=4> [0])>, Input>, %arg1: !spv.ptr<!spv.struct<(!spv.array<128 x f32, stride=4> [0])>, Output>) -> () "None" {
    spv.Return
  }
}

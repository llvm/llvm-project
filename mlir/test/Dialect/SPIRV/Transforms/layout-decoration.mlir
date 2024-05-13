// RUN: mlir-opt -decorate-spirv-composite-type-layout -split-input-file -verify-diagnostics %s -o - | FileCheck %s

spirv.module Logical GLSL450 {
  // CHECK: spirv.GlobalVariable @var0 bind(0, 1) : !spirv.ptr<!spirv.struct<(i32 [0], !spirv.struct<(f32 [0], i32 [4])> [4], f32 [12])>, Uniform>
  spirv.GlobalVariable @var0 bind(0,1) : !spirv.ptr<!spirv.struct<(i32, !spirv.struct<(f32, i32)>, f32)>, Uniform>

  // CHECK: spirv.GlobalVariable @var1 bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.array<64 x i32, stride=4> [0], f32 [256])>, StorageBuffer>
  spirv.GlobalVariable @var1 bind(0,2) : !spirv.ptr<!spirv.struct<(!spirv.array<64xi32>, f32)>, StorageBuffer>

  // CHECK: spirv.GlobalVariable @var2 bind(1, 0) : !spirv.ptr<!spirv.struct<(!spirv.struct<(!spirv.array<64 x i32, stride=4> [0], f32 [256])> [0], i32 [260])>, StorageBuffer>
  spirv.GlobalVariable @var2 bind(1,0) : !spirv.ptr<!spirv.struct<(!spirv.struct<(!spirv.array<64xi32>, f32)>, i32)>, StorageBuffer>

  // CHECK: spirv.GlobalVariable @var3 : !spirv.ptr<!spirv.struct<(!spirv.array<16 x !spirv.struct<(f32 [0], f32 [4], !spirv.array<16 x f32, stride=4> [8])>, stride=72> [0], f32 [1152])>, StorageBuffer>
  spirv.GlobalVariable @var3 : !spirv.ptr<!spirv.struct<(!spirv.array<16x!spirv.struct<(f32, f32, !spirv.array<16xf32>)>>, f32)>, StorageBuffer>

  // CHECK: spirv.GlobalVariable @var4 bind(1, 2) : !spirv.ptr<!spirv.struct<(!spirv.struct<(!spirv.struct<(i1 [0], i8 [1], i16 [2], i32 [4], i64 [8])> [0], f32 [16], i1 [20])> [0], i1 [24])>, StorageBuffer>
  spirv.GlobalVariable @var4 bind(1,2) : !spirv.ptr<!spirv.struct<(!spirv.struct<(!spirv.struct<(i1, i8, i16, i32, i64)>, f32, i1)>, i1)>, StorageBuffer>

  // CHECK: spirv.GlobalVariable @var5 bind(1, 3) : !spirv.ptr<!spirv.struct<(!spirv.array<256 x f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var5 bind(1,3) : !spirv.ptr<!spirv.struct<(!spirv.array<256xf32>)>, StorageBuffer>

  spirv.func @kernel() -> () "None" {
    %c0 = spirv.Constant 0 : i32
    // CHECK: {{%.*}} = spirv.mlir.addressof @var0 : !spirv.ptr<!spirv.struct<(i32 [0], !spirv.struct<(f32 [0], i32 [4])> [4], f32 [12])>, Uniform>
    %0 = spirv.mlir.addressof @var0 : !spirv.ptr<!spirv.struct<(i32, !spirv.struct<(f32, i32)>, f32)>, Uniform>
    // CHECK:  {{%.*}} = spirv.AccessChain {{%.*}}[{{%.*}}] : !spirv.ptr<!spirv.struct<(i32 [0], !spirv.struct<(f32 [0], i32 [4])> [4], f32 [12])>, Uniform>
    %1 = spirv.AccessChain %0[%c0] : !spirv.ptr<!spirv.struct<(i32, !spirv.struct<(f32, i32)>, f32)>, Uniform>, i32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  // CHECK: spirv.GlobalVariable @var0 : !spirv.ptr<!spirv.struct<(!spirv.struct<(!spirv.struct<(!spirv.struct<(!spirv.struct<(i1 [0], i1 [1], f64 [8])> [0], i1 [16])> [0], i1 [24])> [0], i1 [32])> [0], i1 [40])>, Uniform>
  spirv.GlobalVariable @var0 : !spirv.ptr<!spirv.struct<(!spirv.struct<(!spirv.struct<(!spirv.struct<(!spirv.struct<(i1, i1, f64)>, i1)>, i1)>, i1)>, i1)>, Uniform>

  // CHECK: spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.struct<(!spirv.struct<(i16 [0], !spirv.struct<(i1 [0], f64 [8])> [8], f32 [24])> [0], f32 [32])>, Uniform>
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.struct<(!spirv.struct<(i16, !spirv.struct<(i1, f64)>, f32)>, f32)>, Uniform>

  // CHECK: spirv.GlobalVariable @var2 : !spirv.ptr<!spirv.struct<(!spirv.struct<(i16 [0], !spirv.struct<(i1 [0], !spirv.array<16 x !spirv.array<16 x i64, stride=8>, stride=128> [8])> [8], f32 [2064])> [0], f32 [2072])>, Uniform>
  spirv.GlobalVariable @var2 : !spirv.ptr<!spirv.struct<(!spirv.struct<(i16, !spirv.struct<(i1, !spirv.array<16x!spirv.array<16xi64>>)>, f32)>, f32)>, Uniform>

  // CHECK: spirv.GlobalVariable @var3 : !spirv.ptr<!spirv.struct<(!spirv.struct<(!spirv.array<64 x i64, stride=8> [0], i1 [512])> [0], i1 [520])>, Uniform>
  spirv.GlobalVariable @var3 : !spirv.ptr<!spirv.struct<(!spirv.struct<(!spirv.array<64xi64>, i1)>, i1)>, Uniform>

  // CHECK: spirv.GlobalVariable @var4 : !spirv.ptr<!spirv.struct<(i1 [0], !spirv.struct<(i64 [0], i1 [8], i1 [9], i1 [10], i1 [11])> [8], i1 [24])>, Uniform>
  spirv.GlobalVariable @var4 : !spirv.ptr<!spirv.struct<(i1, !spirv.struct<(i64, i1, i1, i1, i1)>, i1)>, Uniform>

  // CHECK: spirv.GlobalVariable @var5 : !spirv.ptr<!spirv.struct<(i1 [0], !spirv.struct<(i1 [0], i1 [1], i1 [2], i1 [3], i64 [8])> [8], i1 [24])>, Uniform>
  spirv.GlobalVariable @var5 : !spirv.ptr<!spirv.struct<(i1, !spirv.struct<(i1, i1, i1, i1, i64)>, i1)>, Uniform>

  // CHECK: spirv.GlobalVariable @var6 : !spirv.ptr<!spirv.struct<(i1 [0], !spirv.struct<(i64 [0], i32 [8], i16 [12], i8 [14], i1 [15])> [8], i1 [24])>, Uniform>
  spirv.GlobalVariable @var6 : !spirv.ptr<!spirv.struct<(i1, !spirv.struct<(i64, i32, i16, i8, i1)>, i1)>, Uniform>

  // CHECK: spirv.GlobalVariable @var7 : !spirv.ptr<!spirv.struct<(i1 [0], !spirv.struct<(!spirv.struct<(i1 [0], i64 [8])> [0], i1 [16])> [8], i1 [32])>, Uniform>
  spirv.GlobalVariable @var7 : !spirv.ptr<!spirv.struct<(i1, !spirv.struct<(!spirv.struct<(i1, i64)>, i1)>, i1)>, Uniform>
}

// -----

spirv.module Logical GLSL450 {
  // CHECK: spirv.GlobalVariable @var0 : !spirv.ptr<!spirv.struct<(vector<2xi32> [0], f32 [8])>, StorageBuffer>
  spirv.GlobalVariable @var0 : !spirv.ptr<!spirv.struct<(vector<2xi32>, f32)>, StorageBuffer>

  // CHECK: spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.struct<(vector<3xi32> [0], f32 [12])>, StorageBuffer>
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.struct<(vector<3xi32>, f32)>, StorageBuffer>

  // CHECK: spirv.GlobalVariable @var2 : !spirv.ptr<!spirv.struct<(vector<4xi32> [0], f32 [16])>, StorageBuffer>
  spirv.GlobalVariable @var2 : !spirv.ptr<!spirv.struct<(vector<4xi32>, f32)>, StorageBuffer>
}

// -----

spirv.module Logical GLSL450 {
  // CHECK: spirv.GlobalVariable @emptyStructAsMember : !spirv.ptr<!spirv.struct<(!spirv.struct<()> [0])>, StorageBuffer>
  spirv.GlobalVariable @emptyStructAsMember : !spirv.ptr<!spirv.struct<(!spirv.struct<()>)>, StorageBuffer>

  // CHECK: spirv.GlobalVariable @arrayType : !spirv.ptr<!spirv.array<4 x !spirv.array<4 x f32>>, StorageBuffer>
  spirv.GlobalVariable @arrayType : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, StorageBuffer>

  // CHECK: spirv.GlobalVariable @InputStorage : !spirv.ptr<!spirv.struct<(!spirv.array<256 x f32>)>, Input>
  spirv.GlobalVariable @InputStorage : !spirv.ptr<!spirv.struct<(!spirv.array<256xf32>)>, Input>

  // CHECK: spirv.GlobalVariable @customLayout : !spirv.ptr<!spirv.struct<(f32 [256], i32 [512])>, Uniform>
  spirv.GlobalVariable @customLayout : !spirv.ptr<!spirv.struct<(f32 [256], i32 [512])>, Uniform>

  // CHECK:  spirv.GlobalVariable @emptyStruct : !spirv.ptr<!spirv.struct<()>, Uniform>
  spirv.GlobalVariable @emptyStruct : !spirv.ptr<!spirv.struct<()>, Uniform>
}

// -----

spirv.module Logical GLSL450 {
  // CHECK: spirv.GlobalVariable @var0 : !spirv.ptr<!spirv.struct<(i32 [0])>, PushConstant>
  spirv.GlobalVariable @var0 : !spirv.ptr<!spirv.struct<(i32)>, PushConstant>
  // CHECK: spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.struct<(i32 [0])>, PhysicalStorageBuffer>
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.struct<(i32)>, PhysicalStorageBuffer>
}

// -----

spirv.module Physical64 GLSL450 {
  // expected-error @+2 {{failed to decorate (unsuported pointee type: '!spirv.struct<rec, (!spirv.ptr<!spirv.struct<rec>, StorageBuffer>)>')}}
  // expected-error @+1 {{failed to legalize operation 'spirv.GlobalVariable'}}
  spirv.GlobalVariable @recursive:
    !spirv.ptr<!spirv.struct<rec, (!spirv.ptr<!spirv.struct<rec>, StorageBuffer>)>, StorageBuffer>
}

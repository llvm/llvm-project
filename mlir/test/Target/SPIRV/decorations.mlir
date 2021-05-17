// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: location = 0 : i32
  spv.GlobalVariable @var {location = 0 : i32} : !spv.ptr<vector<4xf32>, Input>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: no_perspective
  spv.GlobalVariable @var {no_perspective} : !spv.ptr<vector<4xf32>, Input>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: flat
  spv.GlobalVariable @var {flat} : !spv.ptr<si32, Input>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: aliased
  // CHECK: aliased
  spv.GlobalVariable @var1 bind(0, 0) {aliased} : !spv.ptr<!spv.struct<(!spv.array<4xf32, stride=4>[0])>, StorageBuffer>
  spv.GlobalVariable @var2 bind(0, 0) {aliased} : !spv.ptr<!spv.struct<(vector<4xf32>[0])>, StorageBuffer>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: non_readable
  spv.GlobalVariable @var bind(0, 0) {non_readable} : !spv.ptr<!spv.struct<(!spv.array<4xf32, stride=4>[0])>, StorageBuffer>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: non_writable
  spv.GlobalVariable @var bind(0, 0) {non_writable} : !spv.ptr<!spv.struct<(!spv.array<4xf32, stride=4>[0])>, StorageBuffer>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: restrict
  spv.GlobalVariable @var bind(0, 0) {restrict} : !spv.ptr<!spv.struct<(!spv.array<4xf32, stride=4>[0])>, StorageBuffer>
}


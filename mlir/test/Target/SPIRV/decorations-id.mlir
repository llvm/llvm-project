// RUN: mlir-translate --no-implicit-module --split-input-file --test-spirv-roundtrip %s | FileCheck %s

// Round-trip tests for decorations whose operand is a SPIR-V <id>
// (serialized as OpDecorateId, opcode 332).

// AlignmentId references a specialization constant that supplies the alignment.
spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses, Linkage], []> {
  // CHECK: spirv.SpecConstant @sc_align = 16
  // CHECK: alignment_id = @sc_align
  spirv.SpecConstant @sc_align = 16 : i32
  spirv.GlobalVariable @var {alignment_id = @sc_align} : !spirv.ptr<f32, CrossWorkgroup>
}

// -----

// MaxByteOffsetId references a specialization constant.
spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses, Linkage], []> {
  // CHECK: spirv.SpecConstant @sc_offset = 1024
  // CHECK: max_byte_offset_id = @sc_offset
  spirv.SpecConstant @sc_offset = 1024 : i32
  spirv.GlobalVariable @var {max_byte_offset_id = @sc_offset} : !spirv.ptr<f32, CrossWorkgroup>
}

// -----

// CounterBuffer references another global variable.
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: spirv.GlobalVariable @counter
  // CHECK: counter_buffer = @counter
  spirv.GlobalVariable @counter bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4>[0])>, StorageBuffer>
  spirv.GlobalVariable @var bind(0, 0) {counter_buffer = @counter} : !spirv.ptr<!spirv.struct<(!spirv.array<4 x f32, stride=4>[0])>, StorageBuffer>
}

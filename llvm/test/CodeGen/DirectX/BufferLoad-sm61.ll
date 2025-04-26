; RUN: opt -S -dxil-op-lower %s | FileCheck %s
; Before SM6.2 ByteAddressBuffer and StructuredBuffer lower to bufferLoad.

target triple = "dxil-pc-shadermodel6.1-compute"

; CHECK-LABEL: define void @loadf32_struct
define void @loadf32_struct(i32 %index) {
  %buffer = call target("dx.RawBuffer", float, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA:%.*]] = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle %{{.*}}, i32 %index, i32 0)
  %load = call {float, i1}
      @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_f32_0_0_0t(
          target("dx.RawBuffer", float, 0, 0, 0) %buffer,
          i32 %index,
          i32 0)

  ret void
}

; CHECK-LABEL: define void @loadv4f32_byte
define void @loadv4f32_byte(i32 %offset) {
  %buffer = call target("dx.RawBuffer", i8, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA:%.*]] = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle %{{.*}}, i32 %offset, i32 0)
  %load = call {<4 x float>, i1}
      @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_i8_0_0_0t(
          target("dx.RawBuffer", i8, 0, 0, 0) %buffer,
          i32 %offset,
          i32 0)

  ret void
}

; CHECK-LABEL: define void @loadnested
define void @loadnested(i32 %index) {
  %buffer = call
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATAI32:%.*]] = call %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32 68, %dx.types.Handle %{{.*}}, i32 %index, i32 0)
  %loadi32 = call {i32, i1} @llvm.dx.resource.load.rawbuffer.i32(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 0, 0, 0) %buffer,
      i32 %index, i32 0)

  ; CHECK: [[DATAF32:%.*]] = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle %{{.*}}, i32 %index, i32 4)
  %loadf32 = call {<4 x float>, i1} @llvm.dx.resource.load.rawbuffer.v4f32(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 0, 0, 0) %buffer,
      i32 %index, i32 4)

  ; CHECK: [[DATAF16:%.*]] = call %dx.types.ResRet.f16 @dx.op.bufferLoad.f16(i32 68, %dx.types.Handle %{{.*}}, i32 %index, i32 20)
  %loadf16 = call {<3 x half>, i1} @llvm.dx.resource.load.rawbuffer.v3f16(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 0, 0, 0) %buffer,
      i32 %index, i32 20)

  ret void
}

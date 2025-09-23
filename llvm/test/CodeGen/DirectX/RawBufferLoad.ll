; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @f32_user(float)
declare void @v4f32_user(<4 x float>)
declare void @i32_user(i32)
declare void @v4i32_user(<4 x i32>)
declare void @v3f16_user(<3 x half>)
declare void @v4f64_user(<4 x double>)

; CHECK-LABEL: define void @loadf32_struct
define void @loadf32_struct(i32 %index) {
  %buffer = call target("dx.RawBuffer", float, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATA:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %{{.*}}, i32 %index, i32 0, i8 1, i32 4)
  %load = call {float, i1}
      @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_f32_0_0_0t(
          target("dx.RawBuffer", float, 0, 0, 0) %buffer,
          i32 %index,
          i32 0)
  %data = extractvalue {float, i1} %load, 0

  ; CHECK: [[VAL:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA]], 0
  ; CHECK: call void @f32_user(float [[VAL]])
  call void @f32_user(float %data)

  ret void
}

; CHECK-LABEL: define void @loadf32_byte
define void @loadf32_byte(i32 %offset) {
  %buffer = call target("dx.RawBuffer", i8, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATA:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %{{.*}}, i32 %offset, i32 0, i8 1, i32 4)
  %load = call {float, i1}
      @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_i8_0_0_0t(
          target("dx.RawBuffer", i8, 0, 0, 0) %buffer,
          i32 %offset,
          i32 0)
  %data = extractvalue {float, i1} %load, 0

  ; CHECK: [[VAL:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA]], 0
  ; CHECK: call void @f32_user(float [[VAL]])
  call void @f32_user(float %data)

  ret void
}

; CHECK-LABEL: define void @loadv4f32_struct
define void @loadv4f32_struct(i32 %index) {
  %buffer = call target("dx.RawBuffer", <4 x float>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATA:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %{{.*}}, i32 %index, i32 0, i8 15, i32 4)
  %load = call {<4 x float>, i1}
      @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_v4f32_0_0_0t(
          target("dx.RawBuffer", <4 x float>, 0, 0, 0) %buffer,
          i32 %index,
          i32 0)
  %data = extractvalue {<4 x float>, i1} %load, 0

  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA]], 0
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA]], 1
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA]], 2
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA]], 3
  ; CHECK: insertelement <4 x float> poison
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  ; CHECK: call void @v4f32_user(<4 x float>
  call void @v4f32_user(<4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @loadv4f32_byte
define void @loadv4f32_byte(i32 %offset) {
  %buffer = call target("dx.RawBuffer", i8, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATA:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %{{.*}}, i32 %offset, i32 0, i8 15, i32 4)
  %load = call {<4 x float>, i1}
      @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_i8_0_0_0t(
          target("dx.RawBuffer", i8, 0, 0, 0) %buffer,
          i32 %offset,
          i32 0)
  %data = extractvalue {<4 x float>, i1} %load, 0

  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA]], 0
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA]], 1
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA]], 2
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA]], 3
  ; CHECK: insertelement <4 x float> poison
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  ; CHECK: call void @v4f32_user(<4 x float>
  call void @v4f32_user(<4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @loadelements
define void @loadelements(i32 %index) {
  %buffer = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATAF32:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %{{.*}}, i32 %index, i32 0, i8 15, i32 4)
  %loadf32 = call {<4 x float>, i1}
      @llvm.dx.resource.load.rawbuffer.v4f32(
          target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0, 0) %buffer,
          i32 %index,
          i32 0)
  %dataf32 = extractvalue {<4 x float>, i1} %loadf32, 0

  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATAF32]], 0
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATAF32]], 1
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATAF32]], 2
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATAF32]], 3
  ; CHECK: insertelement <4 x float> poison
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  ; CHECK: call void @v4f32_user(<4 x float>
  call void @v4f32_user(<4 x float> %dataf32)

  ; CHECK: [[DATAI32:%.*]] = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %{{.*}}, i32 %index, i32 1, i8 15, i32 4)
  %loadi32 = call {<4 x i32>, i1}
      @llvm.dx.resource.load.rawbuffer.v4i32(
          target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0, 0) %buffer,
          i32 %index,
          i32 1)
  %datai32 = extractvalue {<4 x i32>, i1} %loadi32, 0

  ; CHECK: extractvalue %dx.types.ResRet.i32 [[DATAI32]], 0
  ; CHECK: extractvalue %dx.types.ResRet.i32 [[DATAI32]], 1
  ; CHECK: extractvalue %dx.types.ResRet.i32 [[DATAI32]], 2
  ; CHECK: extractvalue %dx.types.ResRet.i32 [[DATAI32]], 3
  ; CHECK: insertelement <4 x i32> poison
  ; CHECK: insertelement <4 x i32>
  ; CHECK: insertelement <4 x i32>
  ; CHECK: insertelement <4 x i32>
  ; CHECK: call void @v4i32_user(<4 x i32>
  call void @v4i32_user(<4 x i32> %datai32)

  ret void
}

; CHECK-LABEL: define void @loadnested
define void @loadnested(i32 %index) {
  %buffer = call
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATAI32:%.*]] = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %{{.*}}, i32 %index, i32 0, i8 1, i32 4)
  %loadi32 = call {i32, i1} @llvm.dx.resource.load.rawbuffer.i32(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 0, 0, 0) %buffer,
      i32 %index, i32 0)
  %datai32 = extractvalue {i32, i1} %loadi32, 0

  ; CHECK: [[VALI32:%.*]] = extractvalue %dx.types.ResRet.i32 [[DATAI32]], 0
  ; CHECK: call void @i32_user(i32 [[VALI32]])
  call void @i32_user(i32 %datai32)

  ; CHECK: [[DATAF32:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %{{.*}}, i32 %index, i32 4, i8 15, i32 4)
  %loadf32 = call {<4 x float>, i1} @llvm.dx.resource.load.rawbuffer.v4f32(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 0, 0, 0) %buffer,
      i32 %index, i32 4)
  %dataf32 = extractvalue {<4 x float>, i1} %loadf32, 0

  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATAF32]], 0
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATAF32]], 1
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATAF32]], 2
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATAF32]], 3
  ; CHECK: insertelement <4 x float> poison
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  ; CHECK: call void @v4f32_user(<4 x float>
  call void @v4f32_user(<4 x float> %dataf32)

  ; CHECK: [[DATAF16:%.*]] = call %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32 139, %dx.types.Handle %{{.*}}, i32 %index, i32 20, i8 7, i32 2)
  %loadf16 = call {<3 x half>, i1} @llvm.dx.resource.load.rawbuffer.v3f16(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 0, 0, 0) %buffer,
      i32 %index, i32 20)
  %dataf16 = extractvalue {<3 x half>, i1} %loadf16, 0

  ; CHECK: extractvalue %dx.types.ResRet.f16 [[DATAF16]], 0
  ; CHECK: extractvalue %dx.types.ResRet.f16 [[DATAF16]], 1
  ; CHECK: extractvalue %dx.types.ResRet.f16 [[DATAF16]], 2
  ; CHECK: insertelement <3 x half> poison
  ; CHECK: insertelement <3 x half>
  ; CHECK: insertelement <3 x half>
  ; CHECK: call void @v3f16_user(<3 x half>
  call void @v3f16_user(<3 x half> %dataf16)

  ret void
}

; byteaddressbuf.Load<int64_t4>
; CHECK-LABEL: define void @loadv4f64_byte
define void @loadv4f64_byte(i32 %offset) {
  %buffer = call target("dx.RawBuffer", i8, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATA:%.*]] = call %dx.types.ResRet.f64 @dx.op.rawBufferLoad.f64(i32 139, %dx.types.Handle %{{.*}}, i32 %offset, i32 0, i8 15, i32 8)
  %load = call {<4 x double>, i1} @llvm.dx.resource.load.rawbuffer.v4i64(
      target("dx.RawBuffer", i8, 0, 0, 0) %buffer, i32 %offset, i32 0)
  %data = extractvalue {<4 x double>, i1} %load, 0

  ; CHECK: extractvalue %dx.types.ResRet.f64 [[DATA]], 0
  ; CHECK: extractvalue %dx.types.ResRet.f64 [[DATA]], 1
  ; CHECK: extractvalue %dx.types.ResRet.f64 [[DATA]], 2
  ; CHECK: extractvalue %dx.types.ResRet.f64 [[DATA]], 3
  ; CHECK: insertelement <4 x double> poison
  ; CHECK: insertelement <4 x double>
  ; CHECK: insertelement <4 x double>
  ; CHECK: insertelement <4 x double>
  ; CHECK: call void @v4f64_user(<4 x double>
  call void @v4f64_user(<4 x double> %data)

  ret void
}

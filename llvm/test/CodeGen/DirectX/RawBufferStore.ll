; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define void @storef32_struct
define void @storef32_struct(i32 %index, float %data) {
  %buffer = call target("dx.RawBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %buffer_annot, i32 %index, i32 0, float %data, float undef, float undef, float undef, i8 1, i32 4)
  call void @llvm.dx.resource.store.rawbuffer.f32(
      target("dx.RawBuffer", float, 1, 0, 0) %buffer,
      i32 %index, i32 0, float %data)

  ret void
}

; CHECK-LABEL: define void @storef32_byte
define void @storef32_byte(i32 %offset, float %data) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %buffer_annot, i32 %offset, i32 0, float %data, float undef, float undef, float undef, i8 1, i32 4)
  call void @llvm.dx.resource.store.rawbuffer.f32(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer,
      i32 %offset, i32 0, float %data)

  ret void
}

; CHECK-LABEL: define void @storev4f32_struct
define void @storev4f32_struct(i32 %index, <4 x float> %data) {
  %buffer = call target("dx.RawBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0:%.*]] = extractelement <4 x float> %data, i32 0
  ; CHECK: [[DATA1:%.*]] = extractelement <4 x float> %data, i32 1
  ; CHECK: [[DATA2:%.*]] = extractelement <4 x float> %data, i32 2
  ; CHECK: [[DATA3:%.*]] = extractelement <4 x float> %data, i32 3
  ; CHECK: call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %buffer_annot, i32 %index, i32 0, float [[DATA0]], float [[DATA1]], float [[DATA2]], float [[DATA3]], i8 15, i32 4)
  call void @llvm.dx.resource.store.rawbuffer.v4f32(
      target("dx.RawBuffer", <4 x float>, 1, 0, 0) %buffer,
      i32 %index, i32 0, <4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @storev4f32_byte
define void @storev4f32_byte(i32 %offset, <4 x float> %data) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0:%.*]] = extractelement <4 x float> %data, i32 0
  ; CHECK: [[DATA1:%.*]] = extractelement <4 x float> %data, i32 1
  ; CHECK: [[DATA2:%.*]] = extractelement <4 x float> %data, i32 2
  ; CHECK: [[DATA3:%.*]] = extractelement <4 x float> %data, i32 3
  ; CHECK: call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %buffer_annot, i32 %offset, i32 0, float [[DATA0]], float [[DATA1]], float [[DATA2]], float [[DATA3]], i8 15, i32 4)
  call void @llvm.dx.resource.store.rawbuffer.v4f32(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer,
      i32 %offset, i32 0, <4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @storeelements
define void @storeelements(i32 %index, <4 x float> %data0, <4 x i32> %data1) {
  %buffer = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0_0:%.*]] = extractelement <4 x float> %data0, i32 0
  ; CHECK: [[DATA0_1:%.*]] = extractelement <4 x float> %data0, i32 1
  ; CHECK: [[DATA0_2:%.*]] = extractelement <4 x float> %data0, i32 2
  ; CHECK: [[DATA0_3:%.*]] = extractelement <4 x float> %data0, i32 3
  ; CHECK: call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %buffer_annot, i32 %index, i32 0, float [[DATA0_0]], float [[DATA0_1]], float [[DATA0_2]], float [[DATA0_3]], i8 15, i32 4)
  call void @llvm.dx.resource.store.rawbuffer.v4f32(
      target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 1, 0, 0) %buffer,
      i32 %index, i32 0, <4 x float> %data0)

  ; CHECK: [[DATA1_0:%.*]] = extractelement <4 x i32> %data1, i32 0
  ; CHECK: [[DATA1_1:%.*]] = extractelement <4 x i32> %data1, i32 1
  ; CHECK: [[DATA1_2:%.*]] = extractelement <4 x i32> %data1, i32 2
  ; CHECK: [[DATA1_3:%.*]] = extractelement <4 x i32> %data1, i32 3
  ; CHECK: call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %buffer_annot, i32 %index, i32 16, i32 [[DATA1_0]], i32 [[DATA1_1]], i32 [[DATA1_2]], i32 [[DATA1_3]], i8 15, i32 4)
  call void @llvm.dx.resource.store.rawbuffer.v4i32(
      target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 1, 0, 0) %buffer,
      i32 %index, i32 16, <4 x i32> %data1)

  ret void
}

; CHECK-LABEL: define void @storenested
define void @storenested(i32 %index, i32 %data0, <4 x float> %data1, <3 x half> %data2) {
  %buffer = call
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %buffer_annot, i32 %index, i32 0, i32 %data0, i32 undef, i32 undef, i32 undef, i8 1, i32 4)
  call void @llvm.dx.resource.store.rawbuffer.i32(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 1, 0, 0) %buffer,
      i32 %index, i32 0, i32 %data0)

  ; CHECK: [[DATA1_0:%.*]] = extractelement <4 x float> %data1, i32 0
  ; CHECK: [[DATA1_1:%.*]] = extractelement <4 x float> %data1, i32 1
  ; CHECK: [[DATA1_2:%.*]] = extractelement <4 x float> %data1, i32 2
  ; CHECK: [[DATA1_3:%.*]] = extractelement <4 x float> %data1, i32 3
  ; CHECK: call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %buffer_annot, i32 %index, i32 4, float [[DATA1_0]], float [[DATA1_1]], float [[DATA1_2]], float [[DATA1_3]], i8 15, i32 4)
  call void @llvm.dx.resource.store.rawbuffer.v4f32(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 1, 0, 0) %buffer,
      i32 %index, i32 4, <4 x float> %data1)

  ; CHECK: [[DATA2_0:%.*]] = extractelement <3 x half> %data2, i32 0
  ; CHECK: [[DATA2_1:%.*]] = extractelement <3 x half> %data2, i32 1
  ; CHECK: [[DATA2_2:%.*]] = extractelement <3 x half> %data2, i32 2
  ; CHECK: call void @dx.op.rawBufferStore.f16(i32 140, %dx.types.Handle %buffer_annot, i32 %index, i32 20, half [[DATA2_0]], half [[DATA2_1]], half [[DATA2_2]], half undef, i8 7, i32 2)
  call void @llvm.dx.resource.store.rawbuffer.v3f16(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 1, 0, 0) %buffer,
      i32 %index, i32 20, <3 x half> %data2)

  ret void
}

; byteaddressbuf.Store<int64_t4>
; CHECK-LABEL: define void @storev4f64_byte
define void @storev4f64_byte(i32 %offset, <4 x double> %data) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0:%.*]] = extractelement <4 x double> %data, i32 0
  ; CHECK: [[DATA1:%.*]] = extractelement <4 x double> %data, i32 1
  ; CHECK: [[DATA2:%.*]] = extractelement <4 x double> %data, i32 2
  ; CHECK: [[DATA3:%.*]] = extractelement <4 x double> %data, i32 3
  ; CHECK: call void @dx.op.rawBufferStore.f64(i32 140, %dx.types.Handle %buffer_annot, i32 %offset, i32 0, double [[DATA0]], double [[DATA1]], double [[DATA2]], double [[DATA3]], i8 15, i32 8)
  call void @llvm.dx.resource.store.rawbuffer.v4i64(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer,
      i32 %offset, i32 0, <4 x double> %data)

  ret void
}

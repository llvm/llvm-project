; RUN: opt -S -dxil-resource-access %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define void @storefloat4x2_struct
define void @storefloat4x2_struct(i32 %index, <8 x float> %data) {
  %buffer = call target("dx.RawBuffer", [2 x <4 x float>], 1, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", [2 x <4 x float>], 1, 0) %buffer, i32 %index)

  ; CHECK: %[[DATA1:.*]] = shufflevector <8 x float> %data, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_a2v4f32_1_0t.v4f32(target("dx.RawBuffer", [2 x <4 x float>], 1, 0) %buffer, i32 %index, i32 0, <4 x float> %[[DATA1]])
  ; CHECK: %[[DATA2:.*]] = shufflevector <8 x float> %data, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_a2v4f32_1_0t.v4f32(target("dx.RawBuffer", [2 x <4 x float>], 1, 0) %buffer, i32 %index, i32 16, <4 x float> %[[DATA2]])
  store <8 x float> %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @storefloat4x2_byte
define void @storefloat4x2_byte(i32 %index, <8 x float> %data) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0) %buffer, i32 %index)

  ; CHECK: %[[DATA1:.*]] = shufflevector <8 x float> %data, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i8_1_0t.v4f32(target("dx.RawBuffer", i8, 1, 0) %buffer, i32 %index, i32 poison, <4 x float> %[[DATA1]])
  ; CHECK: %[[DATA2:.*]] = shufflevector <8 x float> %data, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ; CHECK: %[[NEXTINDEX:.*]] = add i32 %index, 16
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i8_1_0t.v4f32(target("dx.RawBuffer", i8, 1, 0) %buffer, i32 %[[NEXTINDEX]], i32 poison, <4 x float> %[[DATA2]])
  store <8 x float> %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @storev7f32
define void @storev7f32(i32 %index, <7 x float> %data) {
  %buffer = call target("dx.RawBuffer", <7 x float>, 1, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", <7 x float>, 1, 0) %buffer, i32 %index)

  ; CHECK: %[[DATA1:.*]] = shufflevector <7 x float> %data, <7 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v7f32_1_0t.v4f32(target("dx.RawBuffer", <7 x float>, 1, 0) %buffer, i32 %index, i32 0, <4 x float> %[[DATA1]])
  ; CHECK: %[[DATA2:.*]] = shufflevector <7 x float> %data, <7 x float> poison, <3 x i32> <i32 4, i32 5, i32 6>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v7f32_1_0t.v3f32(target("dx.RawBuffer", <7 x float>, 1, 0) %buffer, i32 %index, i32 16, <3 x float> %[[DATA2]])
  store <7 x float> %data, ptr %ptr

  ret void
}

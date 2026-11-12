; RUN: opt -S -dxil-resource-access %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @v8f32_user(<8 x float>)
declare void @v7f32_user(<7 x float>)

; CHECK-LABEL: define void @loadfloat4x2_struct
define void @loadfloat4x2_struct(i32 %index) {
  %buffer = call target("dx.RawBuffer", [2 x <4 x float>], 1, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", [2 x <4 x float>], 1, 0) %buffer, i32 %index)

  ; CHECK: %[[LOAD1:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_a2v4f32_1_0t(target("dx.RawBuffer", [2 x <4 x float>], 1, 0) %buffer, i32 %index, i32 0)
  ; CHECK: %[[VAL1:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD1]], 0
  ; CHECK: %[[LOAD2:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_a2v4f32_1_0t(target("dx.RawBuffer", [2 x <4 x float>], 1, 0) %buffer, i32 %index, i32 16)
  ; CHECK: %[[VAL2:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD2]], 0
  ; CHECK: %[[MERGED:.*]] = shufflevector <4 x float> %[[VAL1]], <4 x float> %[[VAL2]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ; CHECK: call void @v8f32_user(<8 x float> %[[MERGED]])
  %data = load <8 x float>, ptr %ptr
  call void @v8f32_user(<8 x float> %data)

  ret void
}

; CHECK-LABEL: define void @loadfloat4x2_byte
define void @loadfloat4x2_byte(i32 %index) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0) %buffer, i32 %index)

  ; CHECK: %[[LOAD1:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %buffer, i32 %index, i32 poison)
  ; CHECK: %[[VAL1:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD1]], 0
  ; CHECK: %[[NEXTINDEX:.*]] = add i32 %index, 16
  ; CHECK: %[[LOAD2:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %buffer, i32 %[[NEXTINDEX]], i32 poison)
  ; CHECK: %[[VAL2:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD2]], 0
  ; CHECK: %[[MERGED:.*]] = shufflevector <4 x float> %[[VAL1]], <4 x float> %[[VAL2]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ; CHECK: call void @v8f32_user(<8 x float> %[[MERGED]])
  %data = load <8 x float>, ptr %ptr
  call void @v8f32_user(<8 x float> %data)

  ret void
}

; CHECK-LABEL: define void @loadfloat7
define void @loadfloat7(i32 %index) {
  %buffer = call target("dx.RawBuffer", <7 x float>, 1, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", <7 x float>, 1, 0) %buffer, i32 %index)

  ; CHECK: %[[LOAD1:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_v7f32_1_0t(target("dx.RawBuffer", <7 x float>, 1, 0) %buffer, i32 %index, i32 0)
  ; CHECK: %[[VAL1:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD1]], 0
  ; CHECK: %[[LOAD2:.*]] = call { <3 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v3f32.tdx.RawBuffer_v7f32_1_0t(target("dx.RawBuffer", <7 x float>, 1, 0) %buffer, i32 %index, i32 16)
  ; CHECK: %[[VAL2:.*]] = extractvalue { <3 x float>, i1 } %[[LOAD2]], 0
  ; CHECK: %[[TMP:.*]] = shufflevector <3 x float> %[[VAL2]], <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  ; CHECK: %[[MERGED:.*]] = shufflevector <4 x float> %[[VAL1]], <4 x float> %[[TMP]], <7 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6>
  ; CHECK: call void @v7f32_user(<7 x float> %[[MERGED]])
  %data = load <7 x float>, ptr %ptr
  call void @v7f32_user(<7 x float> %data)

  ret void
}

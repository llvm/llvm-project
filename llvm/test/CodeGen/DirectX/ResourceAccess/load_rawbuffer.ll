; RUN: opt -S -dxil-resource-access %s | FileCheck %s

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

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", float, 0, 0, 0) %buffer, i32 %index)

  ; CHECK: %[[LOAD:.*]] = call { float, i1 } @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_f32_0_0_0t(target("dx.RawBuffer", float, 0, 0, 0) %buffer, i32 %index, i32 0)
  ; CHECK: %[[VAL:.*]] = extractvalue { float, i1 } %[[LOAD]], 0
  ; CHECK: call void @f32_user(float %[[VAL]])
  %data = load float, ptr %ptr
  call void @f32_user(float %data)

  ret void
}

; CHECK-LABEL: define void @loadf32_byte
define void @loadf32_byte(i32 %offset) {
  %buffer = call target("dx.RawBuffer", i8, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 0, 0, 0) %buffer, i32 %offset)

  ; CHECK: %[[LOAD:.*]] = call { float, i1 } @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_i8_0_0_0t(target("dx.RawBuffer", i8, 0, 0, 0) %buffer, i32 %offset, i32 0)
  ; CHECK: %[[VAL:.*]] = extractvalue { float, i1 } %[[LOAD]], 0
  ; CHECK: call void @f32_user(float %[[VAL]])
  %data = load float, ptr %ptr
  call void @f32_user(float %data)

  ret void
}

; CHECK-LABEL: define void @loadv4f32_struct
define void @loadv4f32_struct(i32 %index) {
  %buffer = call target("dx.RawBuffer", <4 x float>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", <4 x float>, 0, 0, 0) %buffer, i32 %index)

  ; CHECK: %[[LOAD:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_v4f32_0_0_0t(target("dx.RawBuffer", <4 x float>, 0, 0, 0) %buffer, i32 %index, i32 0)
  ; CHECK: %[[VAL:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD]], 0
  ; CHECK: call void @v4f32_user(<4 x float> %[[VAL]])
  %data = load <4 x float>, ptr %ptr
  call void @v4f32_user(<4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @loadv4f32_byte
define void @loadv4f32_byte(i32 %offset) {
  %buffer = call target("dx.RawBuffer", i8, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 0, 0, 0) %buffer, i32 %offset)

  ; CHECK: %[[LOAD:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_i8_0_0_0t(target("dx.RawBuffer", i8, 0, 0, 0) %buffer, i32 %offset, i32 0)
  ; CHECK: %[[VAL:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD]], 0
  ; CHECK: call void @v4f32_user(<4 x float> %[[VAL]]
  %data = load <4 x float>, ptr %ptr
  call void @v4f32_user(<4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @loadelements
define void @loadelements(i32 %index) {
  %buffer = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0, 0) %buffer,
      i32 %index)

  ; CHECK: %[[LOADF32:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_sl_v4f32v4i32s_0_0_0t(target("dx.RawBuffer", { <4 x float>, <4 x i32> }, 0, 0, 0) %buffer, i32 %index, i32 0)
  ; CHECK: %[[VALF32:.*]] = extractvalue { <4 x float>, i1 } %[[LOADF32]], 0
  ; CHECK: call void @v4f32_user(<4 x float> %[[VALF32]]
  %dataf32 = load <4 x float>, ptr %ptr
  call void @v4f32_user(<4 x float> %dataf32)

  ; CHECK: %[[LOADI32:.*]] = call { <4 x i32>, i1 } @llvm.dx.resource.load.rawbuffer.v4i32.tdx.RawBuffer_sl_v4f32v4i32s_0_0_0t(target("dx.RawBuffer", { <4 x float>, <4 x i32> }, 0, 0, 0) %buffer, i32 %index, i32 16)
  ; CHECK: %[[VALI32:.*]] = extractvalue { <4 x i32>, i1 } %[[LOADI32]], 0
  ; CHECK: call void @v4i32_user(<4 x i32> %[[VALI32]]
  %addri32 = getelementptr inbounds nuw i8, ptr %ptr, i32 16
  %datai32 = load <4 x i32>, ptr %addri32
  call void @v4i32_user(<4 x i32> %datai32)

  ret void
}

; CHECK-LABEL: define void @loadnested
define void @loadnested(i32 %index) {
  %buffer = call
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 0, 0, 0) %buffer,
      i32 %index)

  ; CHECK: %[[LOADI32:.*]] = call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_sl_i32sl_v4f32v3f16ss_0_0_0t(target("dx.RawBuffer", { i32, { <4 x float>, <3 x half> } }, 0, 0, 0) %buffer, i32 %index, i32 0)
  ; CHECK: %[[VALI32:.*]] = extractvalue { i32, i1 } %[[LOADI32]], 0
  ; CHECK: call void @i32_user(i32 %[[VALI32]])
  %datai32 = load i32, ptr %ptr
  call void @i32_user(i32 %datai32)

  ; CHECK: %[[LOADF32:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_sl_i32sl_v4f32v3f16ss_0_0_0t(target("dx.RawBuffer", { i32, { <4 x float>, <3 x half> } }, 0, 0, 0) %buffer, i32 %index, i32 4)
  ; CHECK: %[[VALF32:.*]] = extractvalue { <4 x float>, i1 } %[[LOADF32]], 0
  ; CHECK: call void @v4f32_user(<4 x float> %[[VALF32]])
  %addrf32 = getelementptr inbounds nuw i8, ptr %ptr, i32 4
  %dataf32 = load <4 x float>, ptr %addrf32
  call void @v4f32_user(<4 x float> %dataf32)

  ; CHECK: %[[LOADF16:.*]] = call { <3 x half>, i1 } @llvm.dx.resource.load.rawbuffer.v3f16.tdx.RawBuffer_sl_i32sl_v4f32v3f16ss_0_0_0t(target("dx.RawBuffer", { i32, { <4 x float>, <3 x half> } }, 0, 0, 0) %buffer, i32 %index, i32 20)
  ; CHECK: %[[VALF16:.*]] = extractvalue { <3 x half>, i1 } %[[LOADF16]], 0
  ; CHECK: call void @v3f16_user(<3 x half> %[[VALF16]])
  %addrf16 = getelementptr inbounds nuw i8, ptr %ptr, i32 20
  %dataf16 = load <3 x half>, ptr %addrf16
  call void @v3f16_user(<3 x half> %dataf16)

  ret void
}

; byteaddressbuf.Load<int64_t4>
; CHECK-LABEL: define void @loadv4f64_byte
define void @loadv4f64_byte(i32 %offset) {
  %buffer = call target("dx.RawBuffer", i8, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 0, 0, 0) %buffer, i32 %offset)

  ; CHECK: %[[LOAD:.*]] = call { <4 x double>, i1 } @llvm.dx.resource.load.rawbuffer.v4f64.tdx.RawBuffer_i8_0_0_0t(target("dx.RawBuffer", i8, 0, 0, 0) %buffer, i32 %offset, i32 0)
  ; CHECK: %[[VAL:.*]] = extractvalue { <4 x double>, i1 } %[[LOAD]], 0
  ; CHECK: call void @v4f64_user(<4 x double> %[[VAL]])
  %data = load <4 x double>, ptr %ptr
  call void @v4f64_user(<4 x double> %data)

  ret void
}

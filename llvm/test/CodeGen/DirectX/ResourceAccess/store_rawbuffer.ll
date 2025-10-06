; RUN: opt -S -dxil-resource-access %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define void @storef32_struct
define void @storef32_struct(i32 %index, float %data) {
  %buffer = call target("dx.RawBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", float, 1, 0, 0) %buffer, i32 %index)

  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_f32_1_0_0t.f32(target("dx.RawBuffer", float, 1, 0, 0) %buffer, i32 %index, i32 0, float %data)
  store float %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @storef32_byte
define void @storef32_byte(i32 %offset, float %data) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset)

  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i8_1_0_0t.f32(target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset, i32 0, float %data)
  store float %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @storev4f32_struct
define void @storev4f32_struct(i32 %index, <4 x float> %data) {
  %buffer = call target("dx.RawBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index)

  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4f32_1_0_0t.v4f32(target("dx.RawBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index, i32 0, <4 x float> %data)
  store <4 x float> %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @storev4f32_byte
define void @storev4f32_byte(i32 %offset, <4 x float> %data) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset)

  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i8_1_0_0t.v4f32(target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset, i32 0, <4 x float> %data)
  store <4 x float> %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @storeelements
define void @storeelements(i32 %index, <4 x float> %dataf32, <4 x i32> %datai32) {
  %buffer = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 1, 0, 0) %buffer,
      i32 %index)

  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_sl_v4f32v4i32s_1_0_0t.v4f32(target("dx.RawBuffer", { <4 x float>, <4 x i32> }, 1, 0, 0) %buffer, i32 %index, i32 0, <4 x float> %dataf32)
  store <4 x float> %dataf32, ptr %ptr

  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_sl_v4f32v4i32s_1_0_0t.v4i32(target("dx.RawBuffer", { <4 x float>, <4 x i32> }, 1, 0, 0) %buffer, i32 %index, i32 16, <4 x i32> %datai32)
  %addri32 = getelementptr inbounds nuw i8, ptr %ptr, i32 16
  store <4 x i32> %datai32, ptr %addri32

  ret void
}

; CHECK-LABEL: define void @storenested
define void @storenested(i32 %index, i32 %datai32, <4 x float> %dataf32, <3 x half> %dataf16) {
  %buffer = call
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", {i32, {<4 x float>, <3 x half>}}, 1, 0, 0) %buffer,
      i32 %index)

  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_sl_i32sl_v4f32v3f16ss_1_0_0t.i32(target("dx.RawBuffer", { i32, { <4 x float>, <3 x half> } }, 1, 0, 0) %buffer, i32 %index, i32 0, i32 %datai32)
  store i32 %datai32, ptr %ptr

  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_sl_i32sl_v4f32v3f16ss_1_0_0t.v4f32(target("dx.RawBuffer", { i32, { <4 x float>, <3 x half> } }, 1, 0, 0) %buffer, i32 %index, i32 4, <4 x float> %dataf32)
  %addrf32 = getelementptr inbounds nuw i8, ptr %ptr, i32 4
  store <4 x float> %dataf32, ptr %addrf32

  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_sl_i32sl_v4f32v3f16ss_1_0_0t.v3f16(target("dx.RawBuffer", { i32, { <4 x float>, <3 x half> } }, 1, 0, 0) %buffer, i32 %index, i32 20, <3 x half> %dataf16)
  %addrf16 = getelementptr inbounds nuw i8, ptr %ptr, i32 20
  store <3 x half> %dataf16, ptr %addrf16

  ret void
}

; byteaddressbuf.Store<int64_t4>
; CHECK-LABEL: define void @storev4f64_byte
define void @storev4f64_byte(i32 %offset, <4 x double> %data) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset)

  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i8_1_0_0t.v4f64(target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset, i32 0, <4 x double> %data)
  store <4 x double> %data, ptr %ptr

  ret void
}

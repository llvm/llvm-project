; RUN: opt -S -dxil-resource-access %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.4-compute"

; struct S {
;   double d;
;   uint4 v;
; };
%struct.S = type <{ double, <4 x i32> }>

; struct T {
;   uint x;
;   S s[16];
;   uint y;
; };
%struct.T = type { i32, [16 x %struct.S], i32 }

; CHECK-LABEL: define void @store_offset_i32
define void @store_offset_i32(i32 %idx, i32 %data) {
  %buffer = call target("dx.RawBuffer", %struct.S, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ;; StructuredBuffer<S> In;
  ;; In[0].v[idx] = data;
  ;
  ; CHECK: [[MUL:%.*]] = mul i32 %idx, 4
  ; CHECK: [[ADD:%.*]] = add i32 [[MUL]], 8
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_s_struct.Ss_0_0t.i32(target("dx.RawBuffer", %struct.S, 0, 0) %buffer, i32 0, i32 [[ADD]], i32 %data)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", %struct.S, 0, 0) %buffer, i32 0)
  %s.i = getelementptr inbounds nuw i8, ptr %ptr, i32 8
  %v.i = getelementptr i32, ptr %s.i, i32 %idx
  store i32 %data, ptr %v.i

  ret void
}

; CHECK-LABEL: define void @store_double
define void @store_double(i32 %idx, double %data) {
  %buffer = call target("dx.RawBuffer", <4 x double>, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ;; StructuredBuffer<double4> In;
  ;; In[0][idx] = data;
  ;
  ; CHECK: [[MUL:%.*]] = mul i32 %idx, 8
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4f64_0_0t.f64(target("dx.RawBuffer", <4 x double>, 0, 0) %buffer, i32 0, i32 [[MUL]], double %data)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", <4 x double>, 0, 0) %buffer, i32 0)
  %v.i = getelementptr double, ptr %ptr, i32 %idx
  store double %data, ptr %v.i

  ret void
}

; CHECK-LABEL: define void @store_half
define void @store_half(i32 %idx, half %data) {
  %buffer = call target("dx.RawBuffer", <4 x half>, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ;; StructuredBuffer<half4> In;
  ;; In[0][idx] = data;
  ;
  ; CHECK: [[MUL:%.*]] = mul i32 %idx, 2
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4f16_0_0t.f16(target("dx.RawBuffer", <4 x half>, 0, 0) %buffer, i32 0, i32 [[MUL]], half %data)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", <4 x half>, 0, 0) %buffer, i32 0)
  %v.i = getelementptr half, ptr %ptr, i32 %idx
  store half %data, ptr %v.i

  ret void
}

; CHECK-LABEL: define void @store_nested
define void @store_nested(i32 %idx, i32 %arrayidx, i32 %vecidx, i32 %data) {
  %buffer = call target("dx.RawBuffer", %struct.T, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ;; StructuredBuffer<T> In;
  ;; In[idx].s[arrayidx].v[vecidx];
  ;
  ; CHECK: [[MUL0:%.*]] = mul i32 %vecidx, 4
  ; CHECK: [[ADD0:%.*]] = add i32 [[MUL0]], 1
  ; CHECK: [[MUL1:%.*]] = mul i32 %arrayidx, 24
  ; CHECK: [[ADD1:%.*]] = add i32 [[ADD0]], [[MUL1]]
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_s_struct.Ts_0_0t.i32(target("dx.RawBuffer", %struct.T, 0, 0) %buffer, i32 %idx, i32 [[ADD1]], i32 %data)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", %struct.T, 0, 0) %buffer, i32 %idx)
  %s.i = getelementptr inbounds nuw %struct.S, ptr %ptr, i32 %arrayidx
  %v.i = getelementptr inbounds nuw i8, ptr %s.i, i32 12
  %i = getelementptr i32, ptr %v.i, i32 %vecidx
  store i32 %data, ptr %i

  ret void
}

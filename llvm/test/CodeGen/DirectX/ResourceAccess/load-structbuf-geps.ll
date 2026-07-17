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

declare void @i32_user(i32)
declare void @double_user(double)
declare void @half_user(half)

; CHECK-LABEL: define void @load_offset_i32
define void @load_offset_i32(i32 %idx) {
  %buffer = call target("dx.RawBuffer", %struct.S, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ;; StructuredBuffer<S> In;
  ;; In[0].v[idx];
  ;
  ; Here we use a GEP to access the vector, even though that isn't necessarily
  ; what the clang frontend will codegen.
  ;
  ; CHECK: [[MUL:%.*]] = mul i32 %idx, 4
  ; CHECK: [[ADD:%.*]] = add i32 [[MUL]], 8
  ; CHECK: [[LOAD:%.*]] = call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_s_struct.Ss_0_0t(target("dx.RawBuffer", %struct.S, 0, 0) %buffer, i32 0, i32 [[ADD]])
  ; CHECK: [[VAL:%.*]] = extractvalue { i32, i1 } [[LOAD]], 0
  ; CHECK: call void @i32_user(i32 [[VAL]])
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", %struct.S, 0, 0) %buffer, i32 0)
  %s.i = getelementptr inbounds nuw i8, ptr %ptr, i32 8
  %v.i = getelementptr i32, ptr %s.i, i32 %idx
  %elt = load i32, ptr %v.i
  call void @i32_user(i32 %elt)

  ;; StructuredBuffer<S> In;
  ;; In[0].v[idx];
  ;
  ; This matches clang's codegen, using extractelement rather than a gep.
  ;
  ; CHECK: [[LOAD:%.*]] = call { <4 x i32>, i1 } @llvm.dx.resource.load.rawbuffer.v4i32.tdx.RawBuffer_s_struct.Ss_0_0t(target("dx.RawBuffer", %struct.S, 0, 0) %buffer, i32 0, i32 8)
  ; CHECK: [[VEC:%.*]] = extractvalue { <4 x i32>, i1 } [[LOAD]], 0
  ; CHECK: [[VAL:%.*]] = extractelement <4 x i32> [[VEC]], i32 %idx
  ; CHECK: call void @i32_user(i32 [[VAL]])
  %ptr2 = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", %struct.S, 0, 0) %buffer, i32 0)
  %v2.i = getelementptr inbounds nuw i8, ptr %ptr, i32 8
  %v2 = load <4 x i32>, ptr %v2.i
  %elt2 = extractelement <4 x i32> %v2, i32 %idx
  call void @i32_user(i32 %elt2)

  ret void
}

; CHECK-LABEL: define void @load_double
define void @load_double(i32 %idx) {
  %buffer = call target("dx.RawBuffer", <4 x double>, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ;; StructuredBuffer<double4> In;
  ;; In[0][idx];
  ;
  ; CHECK: [[MUL:%.*]] = mul i32 %idx, 8
  ; CHECK: [[LOAD:%.*]] = call { double, i1 } @llvm.dx.resource.load.rawbuffer.f64.tdx.RawBuffer_v4f64_0_0t(target("dx.RawBuffer", <4 x double>, 0, 0) %buffer, i32 0, i32 [[MUL]])
  ; CHECK: [[VAL:%.*]] = extractvalue { double, i1 } [[LOAD]], 0
  ; CHECK: call void @double_user(double [[VAL]])
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", <4 x double>, 0, 0) %buffer, i32 0)
  %v.i = getelementptr double, ptr %ptr, i32 %idx
  %v = load double, ptr %v.i
  call void @double_user(double %v)

  ret void
}

; CHECK-LABEL: define void @load_half
define void @load_half(i32 %idx) {
  %buffer = call target("dx.RawBuffer", <4 x half>, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ;; StructuredBuffer<half4> In;
  ;; In[0][idx];
  ;
  ; CHECK: [[MUL:%.*]] = mul i32 %idx, 2
  ; CHECK: [[LOAD:%.*]] = call { half, i1 } @llvm.dx.resource.load.rawbuffer.f16.tdx.RawBuffer_v4f16_0_0t(target("dx.RawBuffer", <4 x half>, 0, 0) %buffer, i32 0, i32 [[MUL]])
  ; CHECK: [[VAL:%.*]] = extractvalue { half, i1 } [[LOAD]], 0
  ; CHECK: call void @half_user(half [[VAL]])
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", <4 x half>, 0, 0) %buffer, i32 0)
  %v.i = getelementptr half, ptr %ptr, i32 %idx
  %v = load half, ptr %v.i
  call void @half_user(half %v)

  ret void
}

; CHECK-LABEL: define void @load_nested
define void @load_nested(i32 %idx, i32 %arrayidx, i32 %vecidx) {
  %buffer = call target("dx.RawBuffer", %struct.T, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ;; StructuredBuffer<T> In;
  ;; In[idx].s[arrayidx].v[vecidx];
  ;
  ; CHECK: [[MUL:%.*]] = mul i32 %arrayidx, 24
  ; CHECK: [[ADD:%.*]] = add i32 12, [[MUL]]
  ; CHECK: [[LOAD:%.*]] = call { <4 x i32>, i1 } @llvm.dx.resource.load.rawbuffer.v4i32.tdx.RawBuffer_s_struct.Ts_0_0t(target("dx.RawBuffer", %struct.T, 0, 0) %buffer, i32 %idx, i32 [[ADD]])
  ; CHECK: [[VEC:%.*]] = extractvalue { <4 x i32>, i1 } [[LOAD]], 0
  ; CHECK: [[VAL:%.*]] = extractelement <4 x i32> [[VEC]], i32 %vecidx
  ; CHECK: call void @i32_user(i32 [[VAL]])
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", %struct.T, 0, 0) %buffer, i32 %idx)
  %s.i = getelementptr inbounds nuw %struct.S, ptr %ptr, i32 %arrayidx
  %v.i = getelementptr inbounds nuw i8, ptr %s.i, i32 12
  %v = load <4 x i32>, ptr %v.i
  %elt = extractelement <4 x i32> %v, i32 %vecidx
  call void @i32_user(i32 %elt)

  ret void
}

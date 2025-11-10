; RUN: opt -S -dxil-resource-access -mtriple=dxil %s | FileCheck %s
;
; Tests for indexed types in dynamically indexed arrays in cbuffers.
;
; struct S {
;   float x[2];
;   uint q;
; };
; cbuffer CB : register(b0) {
;   uint32_t3 w[3]; // offset  0, size 12 (+4) * 3
;   S v[3];         // offset 48, size 24 (+8) * 3
; }
%S = type <{ <{ [1 x <{ float, target("dx.Padding", 12) }>], float }>, i32 }>
%__cblayout_CB = type <{
  <{
    [2 x <{ <3 x i32>, target("dx.Padding", 4) }>],
    <3 x i32>
  }>,
  target("dx.Padding", 4),
  <{
    [2 x <{ %S, target("dx.Padding", 8) }>], %S
  }>
}>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_CB) poison

; CHECK: define void @f
define void @f(ptr %dst, i32 %idx) {
entry:
  %CB.cb_h = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", %__cblayout_CB) %CB.cb_h, ptr @CB.cb, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb
  %CB.cb = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb, align 4

  ;; w[2].z
  ;
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 2)
  ; CHECK: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 2
  ; CHECK: store i32 [[X]], ptr %dst
  %w_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 0)
  %w_gep = getelementptr inbounds nuw i8, ptr addrspace(2) %w_ptr, i32 40
  %w_load = load i32, ptr addrspace(2) %w_gep, align 4
  store i32 %w_load, ptr %dst, align 4

  ;; v[2].q
  ;
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 8)
  ; CHECK: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 4
  ; CHECK: store i32 [[X]], ptr [[PTR]]
  %v_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 48)
  %v_gep = getelementptr inbounds nuw i8, ptr addrspace(2) %v_ptr, i32 84
  %v_load = load i32, ptr addrspace(2) %v_gep, align 4
  %v.i = getelementptr inbounds nuw i8, ptr %dst, i32 4
  store i32 %v_load, ptr %v.i, align 4

  ret void
}

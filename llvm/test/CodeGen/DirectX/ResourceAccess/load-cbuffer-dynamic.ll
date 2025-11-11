; RUN: opt -S -dxil-resource-access -mtriple=dxil %s | FileCheck %s
;
; Tests for dynamic indices into arrays in cbuffers.

; cbuffer CB : register(b0) {
;   uint s[10]; // offset   0,  size  4 (+12) * 10
;   uint t[12]; // offset 160,  size  4 (+12) * 12
; }
%__cblayout_CB = type <{ <{ [9 x <{ i32, target("dx.Padding", 12) }>], i32 }>, target("dx.Padding", 12), <{ [11 x <{ i32, target("dx.Padding", 12) }>], i32 }> }>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_CB) poison

; CHECK: define void @f
define void @f(ptr %dst, i32 %idx) {
entry:
  %CB.cb_h = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", %__cblayout_CB) %CB.cb_h, ptr @CB.cb, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb
  %CB.cb = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb, align 4

  ;; s[idx]
  ;
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 %idx)
  ; CHECK: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
  ; CHECK: store i32 [[X]], ptr %dst
  %s_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 0)
  %s_gep = getelementptr <{ i32, target("dx.Padding", 12) }>, ptr addrspace(2) %s_ptr, i32 %idx
  %s_load = load i32, ptr addrspace(2) %s_gep, align 4
  store i32 %s_load, ptr %dst, align 4

  ;; t[idx]
  ;
  ; CHECK: [[T_IDX:%.*]] = add i32 10, %idx
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 [[T_IDX]])
  ; CHECK: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 4
  ; CHECK: store i32 [[X]], ptr [[PTR]]
  %t_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 160)
  %t_gep = getelementptr <{ i32, target("dx.Padding", 12) }>, ptr addrspace(2) %t_ptr, i32 %idx
  %t_load = load i32, ptr addrspace(2) %t_gep, align 4
  %t.i = getelementptr inbounds nuw i8, ptr %dst, i32 4
  store i32 %t_load, ptr %t.i, align 4

  ret void
}

; RUN: opt -S -dxil-resource-access -mtriple=dxil %s | FileCheck %s

; cbuffer CB : register(b0) {
;   float a1[3];
; }
%__cblayout_CB = type <{ [2 x <{ float, [12 x i8] }>], float }>

@CB.cb = global target("dx.CBuffer", %__cblayout_CB) poison

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB.cb_h = call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", %__cblayout_CB) %CB.cb_h, ptr @CB.cb, align 4

  ;; a1[1]
  ;; Note that the valid GEPs of a1 are `0, 0, 0`, `0, 0, 1`, and `0, 1`.
  ;
  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 1)
  ; CHECK: [[X:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
  ; CHECK: store float [[X]], ptr %dst
  %CB.cb = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb, align 8
  %a1_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 0)
  %a1_gep = getelementptr inbounds <{ [2 x <{ float, [12 x i8] }>], float }>, ptr addrspace(2) %a1_ptr, i32 0, i32 0, i32 1
  %a1 = load float, ptr addrspace(2) %a1_gep, align 4
  store float %a1, ptr %dst, align 32

  ret void
}

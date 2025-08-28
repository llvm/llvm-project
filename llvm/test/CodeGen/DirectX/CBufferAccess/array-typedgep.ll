; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

; cbuffer CB : register(b0) {
;   float a1[3];
; }
%__cblayout_CB = type <{ [3 x float] }>

@CB.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 36, 0)) poison
; CHECK: @CB.cb =
; CHECK-NOT: external {{.*}} addrspace(2) global
@a1 = external addrspace(2) global [3 x float], align 4

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 36, 0)) @llvm.dx.resource.handlefrombinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBs_36_0tt(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 36, 0)) %CB.cb_h, ptr @CB.cb, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 1)
  ; CHECK: [[X:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
  ; CHECK: store float [[X]], ptr %dst
  %a1 = load float, ptr addrspace(2) getelementptr inbounds ([3 x float], ptr addrspace(2) @a1, i32 0, i32 1), align 4
  store float %a1, ptr %dst, align 32

  ret void
}

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1}

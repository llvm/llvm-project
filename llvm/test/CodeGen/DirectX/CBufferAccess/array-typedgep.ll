; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

; cbuffer CB : register(b0) {
;   float a1[3];
; }
%__cblayout_CB = type <{
  <{ [2 x <{ float, target("dx.Padding", 12) }>], float }>
}>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_CB) poison
; CHECK: @CB.cb =
; CHECK-NOT: external {{.*}} addrspace(2) global
@a1 = external local_unnamed_addr addrspace(2) global <{ [2 x <{ float, target("dx.Padding", 12) }>], float }>, align 4

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB.cb_h = call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", %__cblayout_CB) %CB.cb_h, ptr @CB.cb, align 4

  ; a1[1] (accessed via typed gep)
  ;
  ; CHECK: [[PTR:%.*]] = call ptr addrspace(2) @llvm.dx.resource.getpointer.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 0)
  ; CHECK: getelementptr inbounds <{ [2 x <{ float, target("dx.Padding", 12) }>], float }>, ptr addrspace(2) [[PTR]], i32 1, i32 0
  %a1 = load float, ptr addrspace(2) getelementptr inbounds (<{ [2 x <{ float, target("dx.Padding", 12) }>], float }>, ptr addrspace(2) @a1, i32 1, i32 0), align 4
  store float %a1, ptr %dst, align 32

  ret void
}

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1}

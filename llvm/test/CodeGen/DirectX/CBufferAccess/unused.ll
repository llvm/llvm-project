; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s
; Check that we correctly ignore cbuffers that were nulled out by optimizations.

%__cblayout_CB = type <{ float }>
@CB.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_CB) poison
@x = external local_unnamed_addr addrspace(2) global float, align 4

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0, !1, !2}

!0 = !{ptr @CB.cb, ptr addrspace(2) @x}
!1 = !{ptr @CB.cb, null}
!2 = !{null, null}

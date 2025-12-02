; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

; cbuffer CB {
;   float a1;     // offset  0, size  4
;   int a2;       // offset  4, size  4
;   bool a3;      // offset  8, size  4
; }
%__cblayout_CB = type <{ float, i32, i32 }>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_CB) poison
; CHECK: @CB.cb =
; CHECK-NOT: external {{.*}} addrspace(2) global
@a1 = external local_unnamed_addr addrspace(2) global float, align 4
@a2 = external local_unnamed_addr addrspace(2) global i32, align 4
@a3 = external local_unnamed_addr addrspace(2) global i32, align 4

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB.cb_h.i.i = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", %__cblayout_CB) %CB.cb_h.i.i, ptr @CB.cb, align 4

  ; CHECK: call ptr addrspace(2) @llvm.dx.resource.getpointer.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 0)
  %a1 = load float, ptr addrspace(2) @a1, align 4
  store float %a1, ptr %dst, align 8

  ; CHECK: call ptr addrspace(2) @llvm.dx.resource.getpointer.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 4)
  %a2 = load i32, ptr addrspace(2) @a2, align 4
  %a2.i = getelementptr inbounds nuw i8, ptr %dst, i32 4
  store i32 %a2, ptr %a2.i, align 4

  ; CHECK: call ptr addrspace(2) @llvm.dx.resource.getpointer.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 8)
  %a3 = load i32, ptr addrspace(2) @a3, align 4, !range !1, !noundef !2
  %a3.i = getelementptr inbounds nuw i8, ptr %dst, i32 8
  store i32 %a3, ptr %a3.i, align 8

  ret void
}

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1, ptr addrspace(2) @a2, ptr addrspace(2) @a3}
!1 = !{i32 0, i32 2}
!2 = !{}

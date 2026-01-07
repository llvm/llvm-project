; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

; cbuffer CB : register(b0) {
;   float a1[3];        // offset   0,  size 4  (+12) * 3
;   double3 a2[2];      // offset   48, size 24  (+8) * 2
;   float16_t a3[2][2]; // offset  112, size  2 (+14) * 4
; }
%__cblayout_CB = type <{
  <{ [2 x <{ float, target("dx.Padding", 12) }>], float }>, target("dx.Padding", 12),
  <{ [1 x <{ <3 x double>, target("dx.Padding", 8) }>], <3 x double> }>, target("dx.Padding", 8),
  <{ [3 x <{ half, target("dx.Padding", 14) }>], half }>, target("dx.Padding", 14)
}>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_CB) poison
; CHECK: @CB.cb =
; CHECK-NOT: external {{.*}} addrspace(2) global
@a1 = external local_unnamed_addr addrspace(2) global <{ [2 x <{ float, target("dx.Padding", 12) }>], float }>, align 4
@a2 = external local_unnamed_addr addrspace(2) global <{ [1 x <{ <3 x double>, target("dx.Padding", 8) }>], <3 x double> }>, align 32
@a3 = external local_unnamed_addr addrspace(2) global <{ [3 x <{ half, target("dx.Padding", 14) }>], half }>, align 2

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB.cb_h.i.i = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", %__cblayout_CB) %CB.cb_h.i.i, ptr @CB.cb, align 4

  ; a1[1]
  ;
  ; CHECK: [[PTR:%.*]] = call ptr addrspace(2) @llvm.dx.resource.getpointer.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 0)
  ; CHECK: getelementptr inbounds nuw i8, ptr addrspace(2) [[PTR]], i32 16
  %a1 = load float, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a1, i32 16), align 4
  store float %a1, ptr %dst, align 32

  ; a2[1]
  ;
  ; CHECK: [[PTR:%.*]] = call ptr addrspace(2) @llvm.dx.resource.getpointer.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 48)
  ; CHECK: getelementptr inbounds nuw i8, ptr addrspace(2) [[PTR]], i32 32
  %a2 = load <3 x double>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a2, i32 32), align 8
  %a2.i = getelementptr inbounds nuw i8, ptr %dst, i32 8
  store <3 x double> %a2, ptr %a2.i, align 32

  ; a3[0][1]
  ;
  ; CHECK: [[PTR:%.*]] = call ptr addrspace(2) @llvm.dx.resource.getpointer.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 112)
  ; CHECK: getelementptr inbounds nuw i8, ptr addrspace(2) [[PTR]], i32 16
  %a3 = load half, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a3, i32 16), align 2
  %a3.i = getelementptr inbounds nuw i8, ptr %dst, i32 32
  store half %a3, ptr %a3.i, align 2

  ret void
}

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1, ptr addrspace(2) @a2, ptr addrspace(2) @a3}

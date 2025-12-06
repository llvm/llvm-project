; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s
;
; Check that two uses of an identical GEP constant expression generates two
; separate getpointer/gep pairs in the output.

; cbuffer CB : register(b0) {
;   float a1[3];
; }
%__cblayout_CB = type <{ [2 x <{ float, [12 x i8] }>], float }>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_CB) poison
; CHECK: @CB.cb =
; CHECK-NOT: external {{.*}} addrspace(2) global
@a1 = external addrspace(2) global <{ [2 x <{ float, [12 x i8] }>], float }>, align 4

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  ; CHECK: [[PTR:%.*]] = call ptr addrspace(2) @llvm.dx.resource.getpointer.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 0)
  ; CHECK: [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(2) [[PTR]], i32 16
  ; CHECK-COUNT-2: load float, ptr addrspace(2) [[GEP]]
  %a1 = load float, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a1, i32 16), align 4
  store float %a1, ptr %dst, align 32

  %a2 = load float, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a1, i32 16), align 4
  store float %a2, ptr %dst, align 32

  ret void
}

; CHECK: define void @g
define void @g(ptr %dst) {
entry:
  ; CHECK: [[PTR1:%.*]] = call ptr addrspace(2) @llvm.dx.resource.getpointer.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 0)
  ; CHECK: [[GEP1:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(2) [[PTR1]], i32 16
  ; CHECK: load float, ptr addrspace(2) [[GEP1]]
  %a1 = load float, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a1, i32 16), align 4
  store float %a1, ptr %dst, align 32
  br label %next

next:
  ; CHECK: [[PTR2:%.*]] = call ptr addrspace(2) @llvm.dx.resource.getpointer.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 0)
  ; CHECK: [[GEP2:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(2) [[PTR2]], i32 16
  ; CHECK: load float, ptr addrspace(2) [[GEP2]]
  %a2 = load float, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a1, i32 16), align 4
  store float %a2, ptr %dst, align 32

  ret void
}

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1}

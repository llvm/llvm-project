; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

; cbuffer CB : register(b0) {
;   float a1[3];
; }
%__cblayout_CB = type <{ [3 x float] }>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 708, 0, 48, 112, 176, 224, 608, 624, 656)) poison
@a1 = external local_unnamed_addr addrspace(2) global [3 x float], align 4

define void @f(ptr %dst) {
entry:
  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 1)
  ; CHECK: [[X:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
  ; CHECK: store float [[X]], ptr %dst
  %a1 = load float, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a1, i32 4), align 4
  store float %a1, ptr %dst, align 32

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 1)
  ; CHECK: [[X:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
  ; CHECK: store float [[X]], ptr %dst
  %a2 = load float, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a1, i32 4), align 4
  store float %a2, ptr %dst, align 32

  ret void
}

!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1}

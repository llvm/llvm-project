; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

; TODO: Remove datalayout.
; This hack forces dxil-compatible alignment of 3-element 32- and 64-bit vectors
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64-v96:32:32-v192:64:64"

; cbuffer CB : register(b0) {
;   float4 a1[4];  // offset   0,  size 16       * 4
;   float a2[2];   // offset  64,  size 4  (+12) * 2
;   double2 a3[2]; // offset  96,  size 16       * 2
%__cblayout_CB = type <{
  [ 4 x <4 x float> ],
  <{ [2 x <{ float, [12 x i8] }>], float }>, [12 x i8],
  [ 2 x <2 x double> ]
}>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_CB) poison
@a1 = external local_unnamed_addr addrspace(2) global [ 4 x <4 x float> ], align 4
@a2 = external local_unnamed_addr addrspace(2) global <{ [2 x <{ float, [12 x i8] }>], float }>, align 4
@a3 = external local_unnamed_addr addrspace(2) global [ 2 x <2 x double> ], align 8

; CHECK: define void @f(
define void @f(ptr %dst) {
entry:
  %CB.cb_h.i.i = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", %__cblayout_CB) %CB.cb_h.i.i, ptr @CB.cb, align 4

  %a1.copy = alloca [4 x <4 x float>], align 4
  %a3.copy = alloca [2 x <2 x double>], align 8

  ; Try copying no elements
; CHECK-NOT: memcpy
  call void @llvm.memcpy.p0.p2.i32(ptr align 4 %a1.copy, ptr addrspace(2) align 4 @a1, i32 0, i1 false)

  ; Try copying only the first element
; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 0)
; CHECK:    [[X:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
; CHECK:    [[Y:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 1
; CHECK:    [[Z:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 2
; CHECK:    [[A:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 3
; CHECK:    [[UPTO0:%.*]] = insertelement <4 x float> poison, float [[X]], i32 0
; CHECK:    [[UPTO1:%.*]] = insertelement <4 x float> [[UPTO0]], float [[Y]], i32 1
; CHECK:    [[UPTO2:%.*]] = insertelement <4 x float> [[UPTO1]], float [[Z]], i32 2
; CHECK:    [[VALUE:%.*]] = insertelement <4 x float> [[UPTO2]], float [[A]], i32 3
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A1_COPY:%.*]], i32 0

; CHECK:    store <4 x float> [[VALUE]], ptr [[DEST]], align 16
  call void @llvm.memcpy.p0.p2.i32(ptr align 4 %a1.copy, ptr addrspace(2) align 4 @a1, i32 16, i1 false)

  ; Try copying the later element
; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 7)
; CHECK:    [[X:%.*]] = extractvalue { double, double } [[LOAD]], 0
; CHECK:    [[Y:%.*]] = extractvalue { double, double } [[LOAD]], 1
; CHECK:    [[UPTO0:%.*]] = insertelement <2 x double> poison, double [[X]], i32 0
; CHECK:    [[VALUE:%.*]] = insertelement <2 x double> [[UPTO0]], double [[Y]], i32 1
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A3_COPY:%.*]], i32 0
; CHECK:    store <2 x double> [[VALUE]], ptr [[DEST]], align 16
  call void @llvm.memcpy.p0.p2.i32(ptr align 32 %a3.copy, ptr addrspace(2) align 32 @a3, i32 16, i1 false)

  ret void
}

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1, ptr addrspace(2) @a2, ptr addrspace(2) @a3}
!1 = !{i32 0, i32 2}
!2 = !{}

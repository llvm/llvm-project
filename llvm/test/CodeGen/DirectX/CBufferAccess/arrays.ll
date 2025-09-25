; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

; cbuffer CB : register(b0) {
;   float a1[3];
;   double3 a2[2];
;   float16_t a3[2][2];
;   uint64_t a4[3];
;   int4 a5[2][3][4];
;   uint16_t a6[1];
;   int64_t a7[2];
;   bool a8[4];
; }
%__cblayout_CB = type <{ [3 x float], [2 x <3 x double>], [2 x [2 x half]], [3 x i64], [2 x [3 x [4 x <4 x i32>]]], [1 x i16], [2 x i64], [4 x i32] }>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 708, 0, 48, 112, 176, 224, 608, 624, 656)) poison
; CHECK: @CB.cb =
; CHECK-NOT: external {{.*}} addrspace(2) global
@a1 = external local_unnamed_addr addrspace(2) global [3 x float], align 4
@a2 = external local_unnamed_addr addrspace(2) global [2 x <3 x double>], align 32
@a3 = external local_unnamed_addr addrspace(2) global [2 x [2 x half]], align 2
@a4 = external local_unnamed_addr addrspace(2) global [3 x i64], align 8
@a5 = external local_unnamed_addr addrspace(2) global [2 x [3 x [4 x <4 x i32>]]], align 16
@a6 = external local_unnamed_addr addrspace(2) global [1 x i16], align 2
@a7 = external local_unnamed_addr addrspace(2) global [2 x i64], align 8
@a8 = external local_unnamed_addr addrspace(2) global [4 x i32], align 4

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB.cb_h.i.i = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 708, 0, 48, 112, 176, 224, 608, 624, 656)) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 708, 0, 48, 112, 176, 224, 608, 624, 656)) %CB.cb_h.i.i, ptr @CB.cb, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 1)
  ; CHECK: [[X:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
  ; CHECK: store float [[X]], ptr %dst
  %a1 = load float, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a1, i32 4), align 4
  store float %a1, ptr %dst, align 32

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 5)
  ; CHECK: [[X:%.*]] = extractvalue { double, double } [[LOAD]], 0
  ; CHECK: [[Y:%.*]] = extractvalue { double, double } [[LOAD]], 1
  ; CHECK: [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 6)
  ; CHECK: [[Z:%.*]] = extractvalue { double, double } [[LOAD]], 0
  ; CHECK: [[VEC0:%.*]] = insertelement <3 x double> poison, double [[X]], i32 0
  ; CHECK: [[VEC1:%.*]] = insertelement <3 x double> [[VEC0]], double [[Y]], i32 1
  ; CHECK: [[VEC2:%.*]] = insertelement <3 x double> [[VEC1]], double [[Z]], i32 2
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 8
  ; CHECK: store <3 x double> [[VEC2]], ptr [[PTR]]
  %a2 = load <3 x double>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a2, i32 32), align 8
  %a2.i = getelementptr inbounds nuw i8, ptr %dst, i32 8
  store <3 x double> %a2, ptr %a2.i, align 32

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { half, half, half, half, half, half, half, half } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 8)
  ; CHECK: [[X:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 32
  ; CHECK: store half [[X]], ptr [[PTR]]
  %a3 = load half, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a3, i32 6), align 2
  %a3.i = getelementptr inbounds nuw i8, ptr %dst, i32 32
  store half %a3, ptr %a3.i, align 2

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 12)
  ; CHECK: [[X:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 40
  ; CHECK: store i64 [[X]], ptr [[PTR]]
  %a4 = load i64, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a4, i32 8), align 8
  %a4.i = getelementptr inbounds nuw i8, ptr %dst, i32 40
  store i64 %a4, ptr %a4.i, align 8

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 26)
  ; CHECK: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
  ; CHECK: [[Y:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
  ; CHECK: [[Z:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 2
  ; CHECK: [[A:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 3
  ; CHECK: [[VEC0:%.*]] = insertelement <4 x i32> poison, i32 [[X]], i32 0
  ; CHECK: [[VEC1:%.*]] = insertelement <4 x i32> [[VEC0]], i32 [[Y]], i32 1
  ; CHECK: [[VEC2:%.*]] = insertelement <4 x i32> [[VEC1]], i32 [[Z]], i32 2
  ; CHECK: [[VEC3:%.*]] = insertelement <4 x i32> [[VEC2]], i32 [[A]], i32 3
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 48
  ; CHECK: store <4 x i32> [[VEC3]], ptr [[PTR]]
  %a5 = load <4 x i32>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a5, i32 272), align 4
  %a5.i = getelementptr inbounds nuw i8, ptr %dst, i32 48
  store <4 x i32> %a5, ptr %a5.i, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i16, i16, i16, i16, i16, i16, i16, i16 } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 38)
  ; CHECK: [[X:%.*]] = extractvalue { i16, i16, i16, i16, i16, i16, i16, i16 } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 64
  ; CHECK: store i16 [[X]], ptr [[PTR]]
  %a6 = load i16, ptr addrspace(2) @a6, align 2
  %a6.i = getelementptr inbounds nuw i8, ptr %dst, i32 64
  store i16 %a6, ptr %a6.i, align 2

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 40)
  ; CHECK: [[X:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 72
  ; CHECK: store i64 [[X]], ptr [[PTR]]
  %a7 = load i64, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a7, i32 8), align 8
  %a7.i = getelementptr inbounds nuw i8, ptr %dst, i32 72
  store i64 %a7, ptr %a7.i, align 8

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 42)
  ; CHECK: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 80
  ; CHECK: store i32 [[X]], ptr [[PTR]]
  %a8 = load i32, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @a8, i32 4), align 4, !range !1, !noundef !2
  %a8.i = getelementptr inbounds nuw i8, ptr %dst, i32 80
  store i32 %a8, ptr %a8.i, align 4

  ret void
}

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1, ptr addrspace(2) @a2, ptr addrspace(2) @a3, ptr addrspace(2) @a4, ptr addrspace(2) @a5, ptr addrspace(2) @a6, ptr addrspace(2) @a7, ptr addrspace(2) @a8}
!1 = !{i32 0, i32 2}
!2 = !{}

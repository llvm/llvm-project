; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

; cbuffer CB {
;   float3 a1;     // offset   0, size 12 (+4)
;   double3 a2;    // offset  16, size 24
;   float16_t2 a3; // offset  40, size  4 (+4)
;   uint64_t3 a4;  // offset  48, size 24 (+8)
;   int4 a5;       // offset  80, size 16
;   uint16_t3 a6;  // offset  96, size  6 (+10)
; };
%__cblayout_CB = type <{ <3 x float>, <3 x double>, <2 x half>, <3 x i64>, <4 x i32>, <3 x i16> }>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 102, 0, 16, 40, 48, 80, 96)) poison
; CHECK: @CB.cb =
; CHECK-NOT: external {{.*}} addrspace(2) global
@a1 = external local_unnamed_addr addrspace(2) global <3 x float>, align 16
@a2 = external local_unnamed_addr addrspace(2) global <3 x double>, align 32
@a3 = external local_unnamed_addr addrspace(2) global <2 x half>, align 4
@a4 = external local_unnamed_addr addrspace(2) global <3 x i64>, align 32
@a5 = external local_unnamed_addr addrspace(2) global <4 x i32>, align 16
@a6 = external local_unnamed_addr addrspace(2) global <3 x i16>, align 8

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB.cb_h.i.i = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 102, 0, 16, 40, 48, 80, 96)) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 102, 0, 16, 40, 48, 80, 96)) %CB.cb_h.i.i, ptr @CB.cb, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 0)
  ; CHECK: [[X:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
  ; CHECK: [[Y:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 1
  ; CHECK: [[Z:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 2
  ; CHECK: [[VEC0:%.*]] = insertelement <3 x float> poison, float [[X]], i32 0
  ; CHECK: [[VEC1:%.*]] = insertelement <3 x float> [[VEC0]], float [[Y]], i32 1
  ; CHECK: [[VEC2:%.*]] = insertelement <3 x float> [[VEC1]], float [[Z]], i32 2
  ; CHECK: store <3 x float> [[VEC2]], ptr %dst
  %a1 = load <3 x float>, ptr addrspace(2) @a1, align 16
  store <3 x float> %a1, ptr %dst, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 1)
  ; CHECK: [[X:%.*]] = extractvalue { double, double } [[LOAD]], 0
  ; CHECK: [[Y:%.*]] = extractvalue { double, double } [[LOAD]], 1
  ; CHECK: [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 2)
  ; CHECK: [[Z:%.*]] = extractvalue { double, double } [[LOAD]], 0
  ; CHECK: [[VEC0:%.*]] = insertelement <3 x double> poison, double [[X]], i32 0
  ; CHECK: [[VEC1:%.*]] = insertelement <3 x double> [[VEC0]], double [[Y]], i32 1
  ; CHECK: [[VEC2:%.*]] = insertelement <3 x double> [[VEC1]], double [[Z]], i32 2
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 16
  ; CHECK: store <3 x double> [[VEC2]], ptr [[PTR]]
  %a2 = load <3 x double>, ptr addrspace(2) @a2, align 32
  %a2.i = getelementptr inbounds nuw i8, ptr %dst, i32 16
  store <3 x double> %a2, ptr %a2.i, align 8

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { half, half, half, half, half, half, half, half } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 2)
  ; CHECK: [[X:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[LOAD]], 4
  ; CHECK: [[Y:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[LOAD]], 5
  ; CHECK: [[VEC0:%.*]] = insertelement <2 x half> poison, half [[X]], i32 0
  ; CHECK: [[VEC1:%.*]] = insertelement <2 x half> [[VEC0]], half [[Y]], i32 1
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 40
  ; CHECK: store <2 x half> [[VEC1]], ptr [[PTR]]
  %a3 = load <2 x half>, ptr addrspace(2) @a3, align 4
  %a3.i = getelementptr inbounds nuw i8, ptr %dst, i32 40
  store <2 x half> %a3, ptr %a3.i, align 2

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 3)
  ; CHECK: [[X:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
  ; CHECK: [[Y:%.*]] = extractvalue { i64, i64 } [[LOAD]], 1
  ; CHECK: [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 4)
  ; CHECK: [[Z:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
  ; CHECK: [[VEC0:%.*]] = insertelement <3 x i64> poison, i64 [[X]], i32 0
  ; CHECK: [[VEC1:%.*]] = insertelement <3 x i64> [[VEC0]], i64 [[Y]], i32 1
  ; CHECK: [[VEC2:%.*]] = insertelement <3 x i64> [[VEC1]], i64 [[Z]], i32 2
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 48
  ; CHECK: store <3 x i64> [[VEC2]], ptr [[PTR]]
  %a4 = load <3 x i64>, ptr addrspace(2) @a4, align 32
  %a4.i = getelementptr inbounds nuw i8, ptr %dst, i32 48
  store <3 x i64> %a4, ptr %a4.i, align 8

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 5)
  ; CHECK: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
  ; CHECK: [[Y:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
  ; CHECK: [[Z:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 2
  ; CHECK: [[A:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 3
  ; CHECK: [[VEC0:%.*]] = insertelement <4 x i32> poison, i32 [[X]], i32 0
  ; CHECK: [[VEC1:%.*]] = insertelement <4 x i32> [[VEC0]], i32 [[Y]], i32 1
  ; CHECK: [[VEC2:%.*]] = insertelement <4 x i32> [[VEC1]], i32 [[Z]], i32 2
  ; CHECK: [[VEC3:%.*]] = insertelement <4 x i32> [[VEC2]], i32 [[A]], i32 3
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 72
  ; CHECK: store <4 x i32> [[VEC3]], ptr [[PTR]]
  %a5 = load <4 x i32>, ptr addrspace(2) @a5, align 16
  %a5.i = getelementptr inbounds nuw i8, ptr %dst, i32 72
  store <4 x i32> %a5, ptr %a5.i, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i16, i16, i16, i16, i16, i16, i16, i16 } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 6)
  ; CHECK: [[X:%.*]] = extractvalue { i16, i16, i16, i16, i16, i16, i16, i16 } [[LOAD]], 0
  ; CHECK: [[Y:%.*]] = extractvalue { i16, i16, i16, i16, i16, i16, i16, i16 } [[LOAD]], 1
  ; CHECK: [[Z:%.*]] = extractvalue { i16, i16, i16, i16, i16, i16, i16, i16 } [[LOAD]], 2
  ; CHECK: [[VEC0:%.*]] = insertelement <3 x i16> poison, i16 [[X]], i32 0
  ; CHECK: [[VEC1:%.*]] = insertelement <3 x i16> [[VEC0]], i16 [[Y]], i32 1
  ; CHECK: [[VEC2:%.*]] = insertelement <3 x i16> [[VEC1]], i16 [[Z]], i32 2
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 88
  ; CHECK: store <3 x i16> [[VEC2]], ptr [[PTR]]
  %a6 = load <3 x i16>, ptr addrspace(2) @a6, align 8
  %a6.i = getelementptr inbounds nuw i8, ptr %dst, i32 88
  store <3 x i16> %a6, ptr %a6.i, align 2

  ret void
}

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1, ptr addrspace(2) @a2, ptr addrspace(2) @a3, ptr addrspace(2) @a4, ptr addrspace(2) @a5, ptr addrspace(2) @a6}

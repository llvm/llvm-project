; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

; cbuffer CB : register(b0) {
;   float a1[3];
;   double3 a2[2];
;   float16_t a3[2][2];
;   uint64_t a4[3];
;   int2 a5[3][2];
;   uint16_t a6[1];
;   int64_t a7[2];
;   bool a8[4];
; }
%__cblayout_CB = type <{ [3 x float], [2 x <3 x double>], [2 x [2 x half]], [3 x i64], [3 x [2 x <2 x i32>]], [1 x i16], [2 x i64], [4 x i32] }>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 708, 0, 48, 112, 176, 224, 272, 288, 320)) poison
@a1 = external local_unnamed_addr addrspace(2) global [3 x float], align 4
@a2 = external local_unnamed_addr addrspace(2) global [2 x <3 x double>], align 32
@a3 = external local_unnamed_addr addrspace(2) global [2 x [2 x half]], align 2
@a4 = external local_unnamed_addr addrspace(2) global [3 x i64], align 8
@a5 = external local_unnamed_addr addrspace(2) global [3 x [2 x <2 x i32>]], align 16
@a6 = external local_unnamed_addr addrspace(2) global [1 x i16], align 2
@a7 = external local_unnamed_addr addrspace(2) global [2 x i64], align 8
@a8 = external local_unnamed_addr addrspace(2) global [4 x i32], align 4

; CHECK: define void @f(
define void @f(ptr %dst) {
entry:
  %CB.cb_h.i.i = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 708, 0, 48, 112, 176, 224, 272, 288, 320)) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 708, 0, 48, 112, 176, 224, 272, 288, 320)) %CB.cb_h.i.i, ptr @CB.cb, align 4

  %a1.copy = alloca [3 x float], align 4
  %a2.copy = alloca [2 x <3 x double>], align 32
  %a3.copy = alloca [2 x [2 x half]], align 2
  %a4.copy = alloca [3 x i64], align 8
  %a5.copy = alloca [3 x [2 x <2 x i32>]], align 16
  %a6.copy = alloca [1 x i16], align 2
  %a7.copy = alloca [2 x i64], align 8
  %a8.copy = alloca [4 x i32], align 4

  ; Try copying no elements
; CHECK-NOT: memcpy
  call void @llvm.memcpy.p0.p2.i32(ptr align 4 %a1.copy, ptr addrspace(2) align 4 @a1, i32 0, i1 false)

  ; Try copying only the first element
; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 0)
; CHECK:    [[X:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A1_COPY:%.*]], i32 0
; CHECK:    store float [[X]], ptr [[DEST]], align 4
  call void @llvm.memcpy.p0.p2.i32(ptr align 4 %a1.copy, ptr addrspace(2) align 4 @a1, i32 4, i1 false)

; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 0)
; CHECK:    [[X:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A1_COPY:%.*]], i32 0
; CHECK:    store float [[X]], ptr [[DEST]], align 4
; CHECK:    [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 1)
; CHECK:    [[Y:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A1_COPY]], i32 4
; CHECK:    store float [[Y]], ptr [[DEST]], align 4
; CHECK:    [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 2)
; CHECK:    [[Z:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A1_COPY]], i32 8
; CHECK:    store float [[Z]], ptr [[DEST]], align 4
  call void @llvm.memcpy.p0.p2.i32(ptr align 4 %a1.copy, ptr addrspace(2) align 4 @a1, i32 12, i1 false)

; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 3)
; CHECK:    [[X:%.*]] = extractvalue { double, double } [[LOAD]], 0
; CHECK:    [[Y:%.*]] = extractvalue { double, double } [[LOAD]], 1
; CHECK:    [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 4)
; CHECK:    [[Z:%.*]] = extractvalue { double, double } [[LOAD]], 0
; CHECK:    [[UPTO0:%.*]] = insertelement <3 x double> poison, double [[X]], i32 0
; CHECK:    [[UPTO1:%.*]] = insertelement <3 x double> [[UPTO0]], double [[Y]], i32 1
; CHECK:    [[UPTO2:%.*]] = insertelement <3 x double> [[UPTO1]], double [[Z]], i32 2
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A2_COPY:%.*]], i32 0
; CHECK:    store <3 x double> [[UPTO2]], ptr [[DEST]], align 8
; CHECK:    [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 5)
; CHECK:    [[X:%.*]] = extractvalue { double, double } [[LOAD]], 0
; CHECK:    [[Y:%.*]] = extractvalue { double, double } [[LOAD]], 1
; CHECK:    [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 6)
; CHECK:    [[Z:%.*]] = extractvalue { double, double } [[LOAD]], 0
; CHECK:    [[UPTO0:%.*]] = insertelement <3 x double> poison, double [[X]], i32 0
; CHECK:    [[UPTO1:%.*]] = insertelement <3 x double> [[UPTO0]], double [[Y]], i32 1
; CHECK:    [[UPTO2:%.*]] = insertelement <3 x double> [[UPTO1]], double [[Z]], i32 2
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A2_COPY]], i32 24
; CHECK:    store <3 x double> [[UPTO2]], ptr [[DEST]], align 8
  call void @llvm.memcpy.p0.p2.i32(ptr align 32 %a2.copy, ptr addrspace(2) align 32 @a2, i32 48, i1 false)

; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { half, half, half, half, half, half, half, half } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 7)
; CHECK:    [[X:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A3_COPY:%.*]], i32 0
; CHECK:    store half [[X]], ptr [[DEST]], align 2
; CHECK:    [[LOAD:%.*]] = call { half, half, half, half, half, half, half, half } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 8)
; CHECK:    [[Y:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A3_COPY]], i32 2
; CHECK:    store half [[Y]], ptr [[DEST]], align 2
; CHECK:    [[LOAD:%.*]] = call { half, half, half, half, half, half, half, half } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 9)
; CHECK:    [[X:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A3_COPY]], i32 4
; CHECK:    store half [[X]], ptr [[DEST]], align 2
; CHECK:    [[LOAD:%.*]] = call { half, half, half, half, half, half, half, half } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 10)
; CHECK:    [[Y:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A3_COPY]], i32 6
; CHECK:    store half [[Y]], ptr [[DEST]], align 2
  call void @llvm.memcpy.p0.p2.i32(ptr align 2 %a3.copy, ptr addrspace(2) align 2 @a3, i32 8, i1 false)

; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 11)
; CHECK:    [[X:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A4_COPY:%.*]], i32 0
; CHECK:    store i64 [[X]], ptr [[DEST]], align 8
; CHECK:    [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 12)
; CHECK:    [[Y:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A4_COPY]], i32 8
; CHECK:    store i64 [[Y]], ptr [[DEST]], align 8
; CHECK:    [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 13)
; CHECK:    [[Z:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A4_COPY]], i32 16
; CHECK:    store i64 [[Z]], ptr [[DEST]], align 8
  call void @llvm.memcpy.p0.p2.i32(ptr align 8 %a4.copy, ptr addrspace(2) align 8 @a4, i32 24, i1 false)

; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 14)
; CHECK:    [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK:    [[Y:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
; CHECK:    [[UPTO0:%.*]] = insertelement <2 x i32> poison, i32 [[X]], i32 0
; CHECK:    [[UPTO1:%.*]] = insertelement <2 x i32> [[UPTO0]], i32 [[Y]], i32 1
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A5_COPY:%.*]], i32 0
; CHECK:    store <2 x i32> [[UPTO1]], ptr [[DEST]], align 8
; CHECK:    [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 15)
; CHECK:    [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK:    [[Y:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
; CHECK:    [[UPTO0:%.*]] = insertelement <2 x i32> poison, i32 [[X]], i32 0
; CHECK:    [[UPTO1:%.*]] = insertelement <2 x i32> [[UPTO0]], i32 [[Y]], i32 1
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A5_COPY]], i32 8
; CHECK:    store <2 x i32> [[UPTO1]], ptr [[DEST]], align 8
; CHECK:    [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 16)
; CHECK:    [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK:    [[Y:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
; CHECK:    [[UPTO0:%.*]] = insertelement <2 x i32> poison, i32 [[X]], i32 0
; CHECK:    [[UPTO1:%.*]] = insertelement <2 x i32> [[UPTO0]], i32 [[Y]], i32 1
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A5_COPY]], i32 16
; CHECK:    store <2 x i32> [[UPTO1]], ptr [[DEST]], align 8
; CHECK:    [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 17)
; CHECK:    [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK:    [[Y:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
; CHECK:    [[UPTO0:%.*]] = insertelement <2 x i32> poison, i32 [[X]], i32 0
; CHECK:    [[UPTO1:%.*]] = insertelement <2 x i32> [[UPTO0]], i32 [[Y]], i32 1
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A5_COPY]], i32 24
; CHECK:    store <2 x i32> [[UPTO1]], ptr [[DEST]], align 8
; CHECK:    [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 18)
; CHECK:    [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK:    [[Y:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
; CHECK:    [[UPTO0:%.*]] = insertelement <2 x i32> poison, i32 [[X]], i32 0
; CHECK:    [[UPTO1:%.*]] = insertelement <2 x i32> [[UPTO0]], i32 [[Y]], i32 1
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A5_COPY]], i32 32
; CHECK:    store <2 x i32> [[UPTO1]], ptr [[DEST]], align 8
; CHECK:    [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 19)
; CHECK:    [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK:    [[Y:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
; CHECK:    [[UPTO0:%.*]] = insertelement <2 x i32> poison, i32 [[X]], i32 0
; CHECK:    [[UPTO1:%.*]] = insertelement <2 x i32> [[UPTO0]], i32 [[Y]], i32 1
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A5_COPY]], i32 40
; CHECK:    store <2 x i32> [[UPTO1]], ptr [[DEST]], align 8
  call void @llvm.memcpy.p0.p2.i32(ptr align 16 %a5.copy, ptr addrspace(2) align 16 @a5, i32 48, i1 false)

; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { i16, i16, i16, i16, i16, i16, i16, i16 } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 17)
; CHECK:    [[X:%.*]] = extractvalue { i16, i16, i16, i16, i16, i16, i16, i16 } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A6_COPY:%.*]], i32 0
; CHECK:    store i16 [[X]], ptr [[DEST]], align 2
  call void @llvm.memcpy.p0.p2.i32(ptr align 2 %a6.copy, ptr addrspace(2) align 2 @a6, i32 2, i1 false)

; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 18)
; CHECK:    [[X:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A7_COPY:%.*]], i32 0
; CHECK:    store i64 [[X]], ptr [[DEST]], align 8
; CHECK:    [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 19)
; CHECK:    [[Y:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A7_COPY]], i32 8
; CHECK:    store i64 [[Y]], ptr [[DEST]], align 8
  call void @llvm.memcpy.p0.p2.i32(ptr align 8 %a7.copy, ptr addrspace(2) align 8 @a7, i32 16, i1 false)

; CHECK:    [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb, align 4
; CHECK:    [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 20)
; CHECK:    [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A8_COPY:%.*]], i32 0
; CHECK:    store i32 [[X]], ptr [[DEST]], align 4
; CHECK:    [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 21)
; CHECK:    [[Y:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A8_COPY]], i32 4
; CHECK:    store i32 [[Y]], ptr [[DEST]], align 4
; CHECK:    [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 22)
; CHECK:    [[Z:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A8_COPY]], i32 8
; CHECK:    store i32 [[Z]], ptr [[DEST]], align 4
; CHECK:    [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 23)
; CHECK:    [[W:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK:    [[DEST:%.*]] = getelementptr inbounds i8, ptr [[A8_COPY]], i32 12
; CHECK:    store i32 [[W]], ptr [[DEST]], align 4
  call void @llvm.memcpy.p0.p2.i32(ptr align 4 %a8.copy, ptr addrspace(2) align 4 @a8, i32 16, i1 false)

  ret void
}

declare void @llvm.memcpy.p0.p2.i32(ptr noalias writeonly captures(none), ptr addrspace(2) noalias readonly captures(none), i32, i1 immarg)

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1, ptr addrspace(2) @a2, ptr addrspace(2) @a3, ptr addrspace(2) @a4, ptr addrspace(2) @a5, ptr addrspace(2) @a6, ptr addrspace(2) @a7, ptr addrspace(2) @a8}
!1 = !{i32 0, i32 2}
!2 = !{}

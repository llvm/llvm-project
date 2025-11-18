; RUN: opt -S -dxil-cbuffer-access -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

; cbuffer CB {
;   float a1;     // offset  0, size  4
;   int a2;       // offset  4, size  4
;   bool a3;      // offset  8, size  4
;   float16_t a4; // offset 12, size  2
;   uint16_t a5;  // offset 14, size  2
;   double a6;    // offset 16, size  8
;   int64_t a7;   // offset 24, size  8
; }
%__cblayout_CB = type <{ float, i32, i32, half, i16, double, i64 }>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 32, 0, 4, 8, 12, 14, 16, 24)) poison
; CHECK: @CB.cb =
; CHECK-NOT: external {{.*}} addrspace(2) global
@a1 = external local_unnamed_addr addrspace(2) global float, align 4
@a2 = external local_unnamed_addr addrspace(2) global i32, align 4
@a3 = external local_unnamed_addr addrspace(2) global i32, align 4
@a4 = external local_unnamed_addr addrspace(2) global half, align 2
@a5 = external local_unnamed_addr addrspace(2) global i16, align 2
@a6 = external local_unnamed_addr addrspace(2) global double, align 8
@a7 = external local_unnamed_addr addrspace(2) global i64, align 8

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB.cb_h.i.i = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 32, 0, 4, 8, 12, 14, 16, 24)) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 32, 0, 4, 8, 12, 14, 16, 24)) %CB.cb_h.i.i, ptr @CB.cb, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 0)
  ; CHECK: [[A1:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
  ; CHECK: store float [[A1]], ptr %dst
  %a1 = load float, ptr addrspace(2) @a1, align 4
  store float %a1, ptr %dst, align 8

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 0)
  ; CHECK: [[A2:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 4
  ; CHECK: store i32 [[A2]], ptr [[PTR]]
  %a2 = load i32, ptr addrspace(2) @a2, align 4
  %a2.i = getelementptr inbounds nuw i8, ptr %dst, i32 4
  store i32 %a2, ptr %a2.i, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 0)
  ; CHECK: [[A3:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 2
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 8
  ; CHECK: store i32 [[A3]], ptr [[PTR]]
  %a3 = load i32, ptr addrspace(2) @a3, align 4, !range !1, !noundef !2
  %a3.i = getelementptr inbounds nuw i8, ptr %dst, i32 8
  store i32 %a3, ptr %a3.i, align 8

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { half, half, half, half, half, half, half, half } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 0)
  ; CHECK: [[A4:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[LOAD]], 6
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 12
  ; CHECK: store half [[A4]], ptr [[PTR]]
  %a4 = load half, ptr addrspace(2) @a4, align 2
  %a4.i = getelementptr inbounds nuw i8, ptr %dst, i32 12
  store half %a4, ptr %a4.i, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i16, i16, i16, i16, i16, i16, i16, i16 } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 0)
  ; CHECK: [[A5:%.*]] = extractvalue { i16, i16, i16, i16, i16, i16, i16, i16 } [[LOAD]], 7
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 14
  ; CHECK: store i16 [[A5]], ptr [[PTR]]
  %a5 = load i16, ptr addrspace(2) @a5, align 2
  %a5.i = getelementptr inbounds nuw i8, ptr %dst, i32 14
  store i16 %a5, ptr %a5.i, align 2

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 1)
  ; CHECK: [[A6:%.*]] = extractvalue { double, double } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 16
  ; CHECK: store double [[A6]], ptr [[PTR]]
  %a6 = load double, ptr addrspace(2) @a6, align 8
  %a6.i = getelementptr inbounds nuw i8, ptr %dst, i32 16
  store double %a6, ptr %a6.i, align 8

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", {{.*}})), ptr @CB.cb
  ; CHECK: [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", {{.*}})) [[CB]], i32 1)
  ; CHECK: [[A7:%.*]] = extractvalue { i64, i64 } [[LOAD]], 1
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 24
  ; CHECK: store i64 [[A7]], ptr [[PTR]]
  %a7 = load i64, ptr addrspace(2) @a7, align 8
  %a7.i = getelementptr inbounds nuw i8, ptr %dst, i32 24
  store i64 %a7, ptr %a7.i, align 8

  ret void
}

; CHECK-NOT: !hlsl.cbs =
!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @a1, ptr addrspace(2) @a2, ptr addrspace(2) @a3, ptr addrspace(2) @a4, ptr addrspace(2) @a5, ptr addrspace(2) @a6, ptr addrspace(2) @a7}
!1 = !{i32 0, i32 2}
!2 = !{}

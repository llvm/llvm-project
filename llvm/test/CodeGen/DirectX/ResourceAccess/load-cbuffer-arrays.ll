; RUN: opt -S -dxil-resource-access -mtriple=dxil %s | FileCheck %s

; cbuffer CB : register(b0) {
;   float a1[3];        // offset   0,  size 4  (+12) * 3
;   double3 a2[2];      // offset   48, size 24  (+8) * 2
;   float16_t a3[2][2]; // offset  112, size  2 (+14) * 4
;   uint64_t a4[3];     // offset  176, size  8  (+8) * 3
;   int4 a5[2][3][4];   // offset  224, size 16       * 24
;   uint16_t a6[1];     // offset  608, size  2 (+14) * 1
;   int64_t a7[2];      // offset  624, size  8  (+8) * 2
;   bool a8[4];         // offset  656, size  4 (+12) * 4
; }
%__cblayout_CB = type <{
  <{ [2 x <{ float, target("dx.Padding", 12) }>], float }>, target("dx.Padding", 12),
  <{ [1 x <{ <3 x double>, target("dx.Padding", 8) }>], <3 x double> }>, target("dx.Padding", 8),
  <{ [3 x <{ half, target("dx.Padding", 14) }>], half }>, target("dx.Padding", 14),
  <{ [2 x <{ i64, target("dx.Padding", 8) }>], i64 }>, target("dx.Padding", 8),
  [24 x <4 x i32>],
  [1 x i16], target("dx.Padding", 14),
  <{ [1 x <{ i64, target("dx.Padding", 8) }>], i64 }>, target("dx.Padding", 8),
  <{ [3 x <{ i32, target("dx.Padding", 12) }>], i32 }>
}>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_CB) poison

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB.cb_h.i.i = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", %__cblayout_CB) %CB.cb_h.i.i, ptr @CB.cb, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb
  %CB.cb = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb, align 4

  ;; a1[1]
  ;
  ; CHECK: [[LOAD:%.*]] = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 1)
  ; CHECK: [[X:%.*]] = extractvalue { float, float, float, float } [[LOAD]], 0
  ; CHECK: store float [[X]], ptr %dst
  %a1_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 0)
  %a1_gep = getelementptr inbounds nuw i8, ptr addrspace(2) %a1_ptr, i32 16
  %a1 = load float, ptr addrspace(2) %a1_gep, align 4
  store float %a1, ptr %dst, align 32

  ;; a2[1]
  ;
  ; CHECK: [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 5)
  ; CHECK: [[X:%.*]] = extractvalue { double, double } [[LOAD]], 0
  ; CHECK: [[Y:%.*]] = extractvalue { double, double } [[LOAD]], 1
  ; CHECK: [[LOAD:%.*]] = call { double, double } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 6)
  ; CHECK: [[Z:%.*]] = extractvalue { double, double } [[LOAD]], 0
  ; CHECK: [[VEC0:%.*]] = insertelement <3 x double> poison, double [[X]], i32 0
  ; CHECK: [[VEC1:%.*]] = insertelement <3 x double> [[VEC0]], double [[Y]], i32 1
  ; CHECK: [[VEC2:%.*]] = insertelement <3 x double> [[VEC1]], double [[Z]], i32 2
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 8
  ; CHECK: store <3 x double> [[VEC2]], ptr [[PTR]]
  %a2_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 48)
  %a2_gep = getelementptr inbounds nuw i8, ptr addrspace(2) %a2_ptr, i32 32
  %a2 = load <3 x double>, ptr addrspace(2) %a2_gep, align 8
  %a2.i = getelementptr inbounds nuw i8, ptr %dst, i32 8
  store <3 x double> %a2, ptr %a2.i, align 32

  ;; a3[0][1]
  ;
  ; CHECK: [[LOAD:%.*]] = call { half, half, half, half, half, half, half, half } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 8)
  ; CHECK: [[X:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 32
  ; CHECK: store half [[X]], ptr [[PTR]]
  %a3_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 112)
  %a3_gep = getelementptr inbounds nuw i8, ptr addrspace(2) %a3_ptr, i32 16
  %a3 = load half, ptr addrspace(2) %a3_gep, align 2
  %a3.i = getelementptr inbounds nuw i8, ptr %dst, i32 32
  store half %a3, ptr %a3.i, align 2

  ;; a4[1]
  ;
  ; CHECK: [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 12)
  ; CHECK: [[X:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 40
  ; CHECK: store i64 [[X]], ptr [[PTR]]
  %a4_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 176)
  %a4_gep = getelementptr inbounds nuw i8, ptr addrspace(2) %a4_ptr, i32 16
  %a4 = load i64, ptr addrspace(2) %a4_gep, align 8
  %a4.i = getelementptr inbounds nuw i8, ptr %dst, i32 40
  store i64 %a4, ptr %a4.i, align 8

  ;; a5[1][0][0]
  ;
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 26)
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
  %a5_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 224)
  %a5_gep = getelementptr inbounds nuw i8, ptr addrspace(2) %a5_ptr, i32 192
  %a5 = load <4 x i32>, ptr addrspace(2) %a5_gep, align 4
  %a5.i = getelementptr inbounds nuw i8, ptr %dst, i32 48
  store <4 x i32> %a5, ptr %a5.i, align 4

  ;; a6[0]
  ;
  ; CHECK: [[LOAD:%.*]] = call { i16, i16, i16, i16, i16, i16, i16, i16 } @llvm.dx.resource.load.cbufferrow.8.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 38)
  ; CHECK: [[X:%.*]] = extractvalue { i16, i16, i16, i16, i16, i16, i16, i16 } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 64
  ; CHECK: store i16 [[X]], ptr [[PTR]]
  %a6_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 608)
  %a6 = load i16, ptr addrspace(2) %a6_ptr, align 2
  %a6.i = getelementptr inbounds nuw i8, ptr %dst, i32 64
  store i16 %a6, ptr %a6.i, align 2

  ;; a7[1]
  ;
  ; CHECK: [[LOAD:%.*]] = call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 40)
  ; CHECK: [[X:%.*]] = extractvalue { i64, i64 } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 72
  ; CHECK: store i64 [[X]], ptr [[PTR]]
  %a7_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 624)
  %a7_gep = getelementptr inbounds nuw i8, ptr addrspace(2) %a7_ptr, i32 16
  %a7 = load i64, ptr addrspace(2) %a7_gep, align 8
  %a7.i = getelementptr inbounds nuw i8, ptr %dst, i32 72
  store i64 %a7, ptr %a7.i, align 8

  ;; a8[1]
  ;
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 42)
  ; CHECK: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
  ; CHECK: [[PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 80
  ; CHECK: store i32 [[X]], ptr [[PTR]]
  %a8_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 656)
  %a8_gep = getelementptr inbounds nuw i8, ptr addrspace(2) %a8_ptr, i32 16
  %a8 = load i32, ptr addrspace(2) %a8_gep, align 4, !range !0, !noundef !1
  %a8.i = getelementptr inbounds nuw i8, ptr %dst, i32 80
  store i32 %a8, ptr %a8.i, align 4

  ret void
}

!0 = !{i32 0, i32 2}
!1 = !{}

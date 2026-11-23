; RUN: opt -S -dxil-resource-access %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-library"

; struct S {
;     float3 f3;
;     int a;
; };

; struct MyConstants {
;     float f;
;     int2 i2;
;     half3 h3;
;     double d;
;     int array[2];
;     float2x2 m;
;     S s;
; };

; ConstantBuffer<MyConstants> CB;

%MyConstants = type <{ float, <2 x i32>, target("dx.Padding", 4), <3 x half>,
  target("dx.Padding", 2), double, <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }>,
  target("dx.Padding", 12), <{ [1 x <{ <2 x float>, target("dx.Padding", 8) }>], <2 x float> }>,
  target("dx.Padding", 8), %S }>

%S = type <{ <3 x float>, i32 }>

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB_handle = call target("dx.CBuffer", %MyConstants) 
        @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s_MyConstantsst(i32 0, i32 5, i32 1, i32 0, ptr null)
  
; CB.f
;
; CHECK: [[CBLOAD0:%.*]] = call { float, float, float, float } 
; CHECK-SAME: @llvm.dx.resource.load.cbufferrow.4.f32.f32.f32.f32.tdx.CBuffer_s_MyConstantsst(
; CHECK-SAME: target("dx.CBuffer", %MyConstants) %CB_handle, i32 0)
; CHECK-NEXT: [[CB_F:%.*]] = extractvalue { float, float, float, float } [[CBLOAD0]], 0
; CHECK-NEXT: store float [[CB_F]], ptr %dst, align 4
  %CB_ptr0 = call noundef align 1 dereferenceable(72) ptr addrspace(2)
      @llvm.dx.resource.getbasepointer.p2.tdx.CBuffer_s_MyConstantsst(target("dx.CBuffer", %MyConstants) %CB_handle)
  %f_ptr = getelementptr inbounds nuw %MyConstants, ptr addrspace(2) %CB_ptr0, i32 0, i32 0
  %f = load float, ptr addrspace(2) %f_ptr, align 4
  store float %f, ptr %dst, align 4

; CB.i2
;
; CHECK: [[CBLOAD1:%.*]] = call { i32, i32, i32, i32 }
; CHECK-SAME: @llvm.dx.resource.load.cbufferrow.4.i32.i32.i32.i32.tdx.CBuffer_s_MyConstantsst(
; CHECK-SAME: target("dx.CBuffer", %MyConstants) %CB_handle, i32 0)
; CHECK-NEXT: [[CB_I2_0:%.*]] = extractvalue { i32, i32, i32, i32 } [[CBLOAD1]], 1
; CHECK-NEXT: [[CB_I2_1:%.*]] = extractvalue { i32, i32, i32, i32 } [[CBLOAD1]], 2
; CHECK-NEXT: [[VEC0:%.*]] = insertelement <2 x i32> poison, i32 [[CB_I2_0]], i32 0
; CHECK-NEXT: [[VEC1:%.*]] = insertelement <2 x i32> [[VEC0]], i32 [[CB_I2_1]], i32 1
; CHECK-NEXT: [[DST_PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 8
; CHECK-NEXT: store <2 x i32> [[VEC1]], ptr [[DST_PTR]], align 4
  %CB_ptr1 = call noundef align 1 dereferenceable(72) ptr addrspace(2)
    @llvm.dx.resource.getbasepointer.p2.tdx.CBuffer_s_MyConstantsst(target("dx.CBuffer", %MyConstants) %CB_handle)
  %i2_ptr = getelementptr inbounds nuw %MyConstants, ptr addrspace(2) %CB_ptr1, i32 0, i32 1
  %i2 = load <2 x i32>, ptr addrspace(2) %i2_ptr, align 4
  %dst1 = getelementptr inbounds nuw i8, ptr %dst, i32 8
  store <2 x i32> %i2, ptr %dst1, align 4

; CB.h3
;
; CHECK: [[CBLOAD2:%.*]] = call { half, half, half, half, half, half, half, half }
; CHECK-SAME: @llvm.dx.resource.load.cbufferrow.8.f16.f16.f16.f16.f16.f16.f16.f16.tdx.CBuffer_s_MyConstantsst(
; CHECK-SAME: target("dx.CBuffer", %MyConstants) %CB_handle, i32 1)
; CHECK-NEXT: [[CB_H3_0:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[CBLOAD2]], 0
; CHECK-NEXT: [[CB_H3_1:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[CBLOAD2]], 1
; CHECK-NEXT: [[CB_H3_2:%.*]] = extractvalue { half, half, half, half, half, half, half, half } [[CBLOAD2]], 2
; CHECK-NEXT: [[VEC0:%.*]] = insertelement <3 x half> poison, half [[CB_H3_0]], i32 0
; CHECK-NEXT: [[VEC1:%.*]] = insertelement <3 x half> [[VEC0]], half [[CB_H3_1]], i32 1
; CHECK-NEXT: [[VEC2:%.*]] = insertelement <3 x half> [[VEC1]], half [[CB_H3_2]], i32 2
; CHECK-NEXT: [[DST_PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 16
; CHECK-NEXT: store <3 x half> [[VEC2]], ptr [[DST_PTR]], align 2
  %CB_ptr2 = call noundef align 1 dereferenceable(72) ptr addrspace(2)
    @llvm.dx.resource.getbasepointer.p2.tdx.CBuffer_s_MyConstantsst(target("dx.CBuffer", %MyConstants) %CB_handle)
  %h3_ptr = getelementptr inbounds nuw %MyConstants, ptr addrspace(2) %CB_ptr2, i32 0, i32 3
  %h3 = load <3 x half>, ptr addrspace(2) %h3_ptr, align 2
  %dst2 = getelementptr inbounds nuw i8, ptr %dst, i32 16
  store <3 x half> %h3, ptr %dst2, align 2

; CB.d
;
; CHECK: [[CBLOAD3:%.*]] = call { double, double }
; CHECK-SAME: @llvm.dx.resource.load.cbufferrow.2.f64.f64.tdx.CBuffer_s_MyConstantsst(
; CHECK-SAME: target("dx.CBuffer", %MyConstants) %CB_handle, i32 1)
; CHECK-NEXT: [[CB_D3:%.*]] = extractvalue { double, double } [[CBLOAD3]], 1
; CHECK-NEXT: [[DST_PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 32
; CHECK-NEXT: store double [[CB_D3]], ptr [[DST_PTR]], align 8
  %CB_ptr3 = call noundef align 1 dereferenceable(72) ptr addrspace(2)
    @llvm.dx.resource.getbasepointer.p2.tdx.CBuffer_s_MyConstantsst(target("dx.CBuffer", %MyConstants) %CB_handle)
  %d_ptr = getelementptr inbounds nuw %MyConstants, ptr addrspace(2) %CB_ptr3, i32 0, i32 5
  %d = load double, ptr addrspace(2) %d_ptr, align 8
  %dst3 = getelementptr inbounds nuw i8, ptr %dst, i32 32
  store double %d, ptr %dst3, align 8

; CB.array[1]
; - reusing %CB_ptr3 from previous case
;
; CHECK: [[CBLOAD4:%.*]] = call { i32, i32, i32, i32 }
; CHECK-SAME: @llvm.dx.resource.load.cbufferrow.4.i32.i32.i32.i32.tdx.CBuffer_s_MyConstantsst(
; CHECK-SAME: target("dx.CBuffer", %MyConstants) %CB_handle, i32 3)
; CHECK-NEXT: [[ARRAY_1:%.*]] = extractvalue { i32, i32, i32, i32 } [[CBLOAD4]], 0
; CHECK-NEXT: [[DST_PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 40
; CHECK-NEXT: store i32 [[ARRAY_1]], ptr [[DST_PTR]], align 4
   %array = getelementptr inbounds nuw %MyConstants, ptr addrspace(2) %CB_ptr3, i32 0, i32 6
   %arraydecay = getelementptr inbounds [2 x i32], ptr addrspace(2) %array, i32 0, i32 0
   %array_1_ptr = getelementptr <{ i32, target("dx.Padding", 12) }>, ptr addrspace(2) %arraydecay, i32 1, i32 0
   %array_1 = load i32, ptr addrspace(2) %array_1_ptr, align 16
   %dst4 = getelementptr inbounds nuw i8, ptr %dst, i32 40
   store i32 %array_1, ptr %dst4, align 4

; CB.m
; - reusing %CB_ptr3 from previous case
;
; CHECK: [[CBLOAD5:%.*]] = call { float, float, float, float }
; CHECK-SAME: @llvm.dx.resource.load.cbufferrow.4.f32.f32.f32.f32.tdx.CBuffer_s_MyConstantsst(
; CHECK-SAME: target("dx.CBuffer", %MyConstants) %CB_handle, i32 4)
; CHECK-NEXT: [[M00:%.*]] = extractvalue { float, float, float, float } %m.load, 0
; CHECK-NEXT: [[M10:%.*]] = extractvalue { float, float, float, float } %m.load, 1
; CHECK-NEXT: [[M01:%.*]] = extractvalue { float, float, float, float } %m.load, 2
; CHECK-NEXT: [[M11:%.*]] = extractvalue { float, float, float, float } %m.load, 3
; CHECK-NEXT: [[MAT0:%.*]] = insertelement <4 x float> poison, float [[M00]], i32 0
; CHECK-NEXT: [[MAT1:%.*]] = insertelement <4 x float> [[MAT0]], float [[M10]], i32 1
; CHECK-NEXT: [[MAT2:%.*]] = insertelement <4 x float> [[MAT1]], float [[M01]], i32 2
; CHECK-NEXT: [[MAT3:%.*]] = insertelement <4 x float> [[MAT2]], float [[M11]], i32 3
; CHECK-NEXT: [[DST_PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 56
; CHECK-NEXT: store <4 x float> [[MAT3]], ptr [[DST_PTR]], align 4
   %m_ptr = getelementptr inbounds nuw %MyConstants, ptr addrspace(2) %CB_ptr3, i32 0, i32 8
   %m = load <4 x float>, ptr addrspace(2) %m_ptr, align 4
   %dst5 = getelementptr inbounds nuw i8, ptr %dst, i32 56
   store <4 x float> %m, ptr %dst5, align 4

; CB.s
; - reusing %CB_ptr3 from previous case

; CHECK-NEXT: [[DST_PTR1:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 72
; CHECK-NEXT: [[CBLOAD6:%.*]] = call { float, float, float, float }
; CHECK-SAME: @llvm.dx.resource.load.cbufferrow.4.f32.f32.f32.f32.tdx.CBuffer_s_MyConstantsst
; CHECK-SAME: target("dx.CBuffer", %MyConstants) %CB_handle, i32 5)
; CHECK-NEXT: [[F3_0:%.*]] = extractvalue { float, float, float, float } [[CBLOAD6]], 0
; CHECK-NEXT: [[F3_1:%.*]] = extractvalue { float, float, float, float } [[CBLOAD6]], 1
; CHECK-NEXT: [[F3_2:%.*]] = extractvalue { float, float, float, float } [[CBLOAD6]], 2
; CHECK-NEXT: [[VEC0:%.*]] = insertelement <3 x float> poison, float [[F3_0]], i32 0
; CHECK-NEXT: [[VEC1:%.*]] = insertelement <3 x float> [[VEC0]], float [[F3_1]], i32 1
; CHECK-NEXT: [[VEC2:%.*]] = insertelement <3 x float> [[VEC1]], float [[F3_2]], i32 2
; CHECK-NEXT: store <3 x float> [[VEC2]], ptr [[DST_PTR1]], align 4
; CHECK-NEXT: [[DST_PTR2:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 84
; CHECK-NEXT: [[CBLOAD7:%.*]] = call { i32, i32, i32, i32 }
; CHECK-SAME: @llvm.dx.resource.load.cbufferrow.4.i32.i32.i32.i32.tdx.CBuffer_s_MyConstantsst
; CHECK-SAME: target("dx.CBuffer", %MyConstants) %CB_handle, i32 5)
; CHECK-NEXT: [[A:%.*]] = extractvalue { i32, i32, i32, i32 } [[CBLOAD7]], 3
; CHECK-NEXT: store i32 [[A]], ptr [[DST_PTR2]], align 4
   %CB_ptr4 = call noundef align 1 dereferenceable(72) ptr addrspace(2)
      @llvm.dx.resource.getbasepointer.p2.tdx.CBuffer_s_MyConstantsst(target("dx.CBuffer", %MyConstants) %CB_handle)
   %s_ptr = getelementptr inbounds nuw %MyConstants, ptr addrspace(2) %CB_ptr4, i32 0, i32 9
   %s_f3_ptr = getelementptr inbounds %S, ptr addrspace(2) %s_ptr, i32 0, i32 0
   %dst6 = getelementptr inbounds nuw i8, ptr %dst, i32 72
   %s_f3  = load <3 x float>, ptr addrspace(2) %s_f3_ptr, align 4
   store <3 x float> %s_f3, ptr %dst6, align 4
   %s_a_ptr = getelementptr inbounds %S, ptr addrspace(2) %s_ptr, i32 0, i32 1
   %dst7 = getelementptr inbounds nuw i8, ptr %dst, i32 84
   %s_a = load i32, ptr addrspace(2) %s_a_ptr, align 4
   store i32 %s_a, ptr %dst7, align 4

; - CB.s.a
;
; CHECK: [[CBLOAD8:%.*]] = call { i32, i32, i32, i32 }
; CHECK-SAME: @llvm.dx.resource.load.cbufferrow.4.i32.i32.i32.i32.tdx.CBuffer_s_MyConstantsst(
; CHECK-SAME: target("dx.CBuffer", %MyConstants) %CB_handle, i32 5)
; CHECK-NEXT: [[S_A:%.*]] = extractvalue { i32, i32, i32, i32 } [[CBLOAD8]], 3
; CHECK-NEXT: [[DST_PTR:%.*]] = getelementptr inbounds nuw i8, ptr %dst, i32 88
; CHECK-NEXT: store i32 [[S_A]], ptr [[DST_PTR]], align 4
  %CB_ptr5 = call noundef align 1 dereferenceable(72) ptr addrspace(2)
      @llvm.dx.resource.getbasepointer.p2.tdx.CBuffer_s_MyConstantsst(target("dx.CBuffer", %MyConstants) %CB_handle)
  %s1_ptr = getelementptr inbounds nuw %MyConstants, ptr addrspace(2) %CB_ptr5, i32 0, i32 9
  %s1_a_ptr = getelementptr inbounds nuw %S, ptr addrspace(2) %s_ptr, i32 0, i32 1
  %s1_a = load i32, ptr addrspace(2) %s1_a_ptr, align 4
  %dst8 = getelementptr inbounds nuw i8, ptr %dst, i32 88
  store i32 %s1_a, ptr %dst8, align 4

  ret void
}

; CHECK-NOT: call {{.*}} @llvm.dx.resource.getbasepointer

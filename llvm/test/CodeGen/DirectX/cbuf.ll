; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s --check-prefix=DXILMD
; RUN: opt -S --passes="dxil-pretty-printer" < %s 2>&1 | FileCheck %s --check-prefix=PRINT

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-library"

; Make sure the size is 24 = 16 + 8 (float,i32,double -> 16 and int2 -> 8)
; DXILMD:!{i32 0, ptr @A.cb., !"", i32 1, i32 2, i32 1, i32 24}

; Make sure match register(b2, space1) with ID 0.
; PRINT:cbuffer      NA          NA     CB0     cb2,space1     1

@A.cb. = external constant { float, i32, double, <2 x i32> }

; Function Attrs: noinline nounwind optnone
define noundef float @"?foo@@YAMXZ"() #0 {
entry:
  %0 = load float, ptr @A.cb., align 4
  %conv = fpext float %0 to double
  %1 = load double, ptr getelementptr inbounds ({ float, i32, double, <2 x i32> }, ptr @A.cb., i32 0, i32 2), align 8
  %2 = load <2 x i32>, ptr getelementptr inbounds ({ float, i32, double, <2 x i32> }, ptr @A.cb., i32 0, i32 3), align 8
  %3 = extractelement <2 x i32> %2, i32 1
  %conv1 = sitofp i32 %3 to double
  %4 = call double @llvm.fmuladd.f64(double %1, double %conv1, double %conv)
  %conv2 = fptrunc double %4 to float
  ret float %conv2
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.fmuladd.f64(double, double, double) #1

attributes #0 = { noinline nounwind }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!hlsl.cbufs = !{!1}

!1 = !{ptr @A.cb., !"A.cb.ty", i32 13, i1 false, i32 2, i32 1}

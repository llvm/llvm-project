; RUN: llc < %s -mtriple=armv6-apple-darwin9 -mattr=+vfp2
; rdar://6653182


%struct.ggBRDF = type { i32 (...)** }
%struct.ggPoint2 = type { [2 x double] }
%struct.ggPoint3 = type { [3 x double] }
%struct.ggSpectrum = type { [8 x float] }
%struct.ggSphere = type { %struct.ggPoint3, double }
%struct.mrDiffuseAreaSphereLuminaire = type { %struct.mrSphere, %struct.ggSpectrum }
%struct.mrDiffuseCosineSphereLuminaire = type { %struct.mrDiffuseAreaSphereLuminaire }
%struct.mrSphere = type { %struct.ggBRDF, %struct.ggSphere }

declare double @llvm.sqrt.f64(double) nounwind readonly

declare double @sin(double) nounwind readonly

declare double @acos(double) nounwind readonly

define i32 @_ZNK34mrDiffuseSolidAngleSphereLuminaire18selectVisiblePointERK8ggPoint3RK9ggVector3RK8ggPoint2dRS0_Rd(%struct.mrDiffuseCosineSphereLuminaire* nocapture %this, %struct.ggPoint3* nocapture %x, %struct.ggPoint3* nocapture %unnamed_arg, %struct.ggPoint2* nocapture %uv, double %unnamed_arg2, %struct.ggPoint3* nocapture %on_light, double* nocapture %invProb) nounwind {
entry:
  %0 = call double @llvm.sqrt.f64(double 0.000000e+00) nounwind
  %1 = fcmp ult double 0.000000e+00, %0
  br i1 %1, label %bb3, label %bb7

bb3:                                              ; preds = %entry
  %2 = fdiv double 1.000000e+00, 0.000000e+00
  %3 = fmul double 0.000000e+00, %2
  %4 = call double @llvm.sqrt.f64(double 0.000000e+00) nounwind
  %5 = fdiv double 1.000000e+00, %4
  %6 = fmul double %3, %5
  %7 = fmul double 0.000000e+00, %5
  %8 = fmul double %3, %7
  %9 = fsub double %8, 0.000000e+00
  %10 = fmul double 0.000000e+00, %6
  %11 = fsub double 0.000000e+00, %10
  %12 = fsub double -0.000000e+00, %11
  %13 = fmul double %0, %0
  %14 = fsub double %13, 0.000000e+00
  %15 = call double @llvm.sqrt.f64(double %14)
  %16 = fmul double 0.000000e+00, %15
  %17 = fdiv double %16, %0
  %18 = fadd double 0.000000e+00, %17
  %19 = call double @acos(double %18) nounwind readonly
  %20 = load double* null, align 4
  %21 = fmul double %20, 0x401921FB54442D18
  %22 = call double @sin(double %19) nounwind readonly
  %23 = fmul double %22, 0.000000e+00
  %24 = fmul double %6, %23
  %25 = fmul double %7, %23
  %26 = call double @sin(double %21) nounwind readonly
  %27 = fmul double %22, %26
  %28 = fmul double %9, %27
  %29 = fmul double %27, %12
  %30 = fadd double %24, %28
  %31 = fadd double 0.000000e+00, %29
  %32 = fadd double %25, 0.000000e+00
  %33 = fadd double %30, 0.000000e+00
  %34 = fadd double %31, 0.000000e+00
  %35 = fadd double %32, 0.000000e+00
  %36 = bitcast %struct.ggPoint3* %x to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* null, i8* %36, i32 24, i32 4, i1 false)
  store double %33, double* null, align 8
  br i1 false, label %_Z20ggRaySphereIntersectRK6ggRay3RK8ggSphereddRd.exit, label %bb5.i.i.i

bb5.i.i.i:                                        ; preds = %bb3
  unreachable

_Z20ggRaySphereIntersectRK6ggRay3RK8ggSphereddRd.exit: ; preds = %bb3
  %37 = fsub double %13, 0.000000e+00
  %38 = fsub double -0.000000e+00, %34
  %39 = fsub double -0.000000e+00, %35
  ret i32 1

bb7:                                              ; preds = %entry
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

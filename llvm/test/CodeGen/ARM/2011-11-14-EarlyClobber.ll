; RUN: llc < %s -mcpu=cortex-a8 -verify-regalloc
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios"

; This test calls shrinkToUses with an early-clobber redefined live range during
; spilling.
;
;   Shrink: %47,1.158257e-02 = [384r,400e:0)[400e,420r:1)  0@384r 1@400e
;
; The early-clobber instruction is an str:
;
;   %12<earlyclobber,def> = t2STR_PRE %6, %12, 32, pred:14, pred:%noreg
;
; This tests that shrinkToUses handles the EC redef correctly.

%struct.Transform_Struct.0.11.12.17.43.46.56.58.60 = type { [4 x [4 x double]] }

define void @Compute_Axis_Rotation_Transform(%struct.Transform_Struct.0.11.12.17.43.46.56.58.60* nocapture %transform, double* nocapture %V1, double %angle) nounwind {
entry:
  store double 1.000000e+00, double* null, align 4
  %arrayidx5.1.i = getelementptr inbounds %struct.Transform_Struct.0.11.12.17.43.46.56.58.60, %struct.Transform_Struct.0.11.12.17.43.46.56.58.60* %transform, i32 0, i32 0, i32 0, i32 1
  store double 0.000000e+00, double* %arrayidx5.1.i, align 4
  %arrayidx5.2.i = getelementptr inbounds %struct.Transform_Struct.0.11.12.17.43.46.56.58.60, %struct.Transform_Struct.0.11.12.17.43.46.56.58.60* %transform, i32 0, i32 0, i32 0, i32 2
  store double 0.000000e+00, double* %arrayidx5.2.i, align 4
  %arrayidx5.114.i = getelementptr inbounds %struct.Transform_Struct.0.11.12.17.43.46.56.58.60, %struct.Transform_Struct.0.11.12.17.43.46.56.58.60* %transform, i32 0, i32 0, i32 1, i32 0
  store double 0.000000e+00, double* %arrayidx5.114.i, align 4
  %arrayidx5.1.1.i = getelementptr inbounds %struct.Transform_Struct.0.11.12.17.43.46.56.58.60, %struct.Transform_Struct.0.11.12.17.43.46.56.58.60* %transform, i32 0, i32 0, i32 1, i32 1
  store double 1.000000e+00, double* %arrayidx5.1.1.i, align 4
  store double 0.000000e+00, double* null, align 4
  store double 1.000000e+00, double* null, align 4
  store double 0.000000e+00, double* null, align 4
  %call = tail call double @cos(double %angle) nounwind readnone
  %call1 = tail call double @sin(double %angle) nounwind readnone
  %0 = load double, double* %V1, align 4
  %arrayidx2 = getelementptr inbounds double, double* %V1, i32 1
  %1 = load double, double* %arrayidx2, align 4
  %mul = fmul double %0, %1
  %sub = fsub double 1.000000e+00, %call
  %mul3 = fmul double %mul, %sub
  %2 = load double, double* undef, align 4
  %mul5 = fmul double %2, %call1
  %add = fadd double %mul3, %mul5
  store double %add, double* %arrayidx5.1.i, align 4
  %3 = load double, double* %V1, align 4
  %mul11 = fmul double %3, undef
  %mul13 = fmul double %mul11, %sub
  %4 = load double, double* %arrayidx2, align 4
  %mul15 = fmul double %4, %call1
  %sub16 = fsub double %mul13, %mul15
  store double %sub16, double* %arrayidx5.2.i, align 4
  %5 = load double, double* %V1, align 4
  %6 = load double, double* %arrayidx2, align 4
  %mul22 = fmul double %5, %6
  %mul24 = fmul double %mul22, %sub
  %sub27 = fsub double %mul24, undef
  store double %sub27, double* %arrayidx5.114.i, align 4
  ret void
}

declare double @cos(double) nounwind readnone

declare double @sin(double) nounwind readnone

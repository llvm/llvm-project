; RUN: llc < %s -mtriple powerpc64le-unknown-linux-gnu
; RUN: llc < %s -mcpu=ppc -mtriple powerpc64-unknown-linux-gnu
; RUN: llc < %s -mcpu=ppc -mtriple powerpc-ibm-aix
; RUN: llc < %s -mcpu=ppc -mtriple powerpc64-ibm-aix

define void @xvcvdpsp_kill_flag() {
entry:
  %call49 = tail call double @sin()
  %0 = insertelement <2 x double> poison, double %call49, i64 1
  %1 = fmul <2 x double> %0, zeroinitializer
  %2 = shufflevector <2 x double> %1, <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %3 = insertelement <4 x double> %2, double 0.000000e+00, i64 2
  %4 = insertelement <4 x double> %3, double poison, i64 3
  %5 = fptrunc <4 x double> %4 to <4 x float>
  %6 = shufflevector <4 x float> %5, <4 x float> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %7 = shufflevector <4 x float> %5, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 3, i32 3>
  %8 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %7, <4 x float> <float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00>, <4 x float> zeroinitializer)
  br label %if.end1

if.end1:                                         ; preds = %entry
  br i1 poison, label %for.cond1.preheader, label %if.then2

for.cond1.preheader:                            ; preds = %if.end1
  br label %for.body2.preheader

for.body2.preheader:                            ; preds = %for.cond1.preheader
  br i1 poison, label %for.loopexit, label %for.body3

for.body3:                                      ; preds = %for.body2.preheader
  %9 = tail call <4 x float> @llvm.ppc.fnmsub.v4f32(<4 x float> zeroinitializer, <4 x float> %6, <4 x float> zeroinitializer)
  %10 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float> %8, <4 x float> %9)
  %11 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %10, <4 x float> zeroinitializer, <4 x float> zeroinitializer)
  store <4 x float> %11, ptr poison, align 16
  unreachable

for.loopexit:                    ; preds = %for.body2.preheader
  unreachable

if.then2:                                       ; preds = %if.end1
  ret void
}

declare double @sin() local_unnamed_addr #0
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.ppc.fnmsub.v4f32(<4 x float>, <4 x float>, <4 x float>)

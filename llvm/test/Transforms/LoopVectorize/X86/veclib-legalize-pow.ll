; RUN: opt -vector-library=SVML -passes=inject-tli-mappings,loop-vectorize \
; RUN:   -force-vector-width=8 -force-vector-interleave=1 -mattr=avx \
; RUN:   -legalize-vector-library-calls -S < %s | FileCheck %s

; Verify legalization of a two-argument function: pow(double, double) at VF=8
; on AVX (256-bit) is split into 2 x VF=4 calls to __svml_pow4, with both
; arguments properly extracted via sub-vector shuffles.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @pow_f64(ptr nocapture %varray, ptr nocapture readonly %exponents) {
; CHECK-LABEL: @pow_f64(
; CHECK:       vector.body:
;
; First part: extract [0..3] from both wide args, call __svml_pow4.
; CHECK:         [[EXT_X0:%.*]] = shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK:         [[EXT_Y0:%.*]] = shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK:         [[CALL0:%.*]] = call fast <4 x double> @__svml_pow4(<4 x double> [[EXT_X0]], <4 x double> [[EXT_Y0]])
;
; Second part: extract [4..7] from both wide args, call __svml_pow4.
; CHECK:         [[EXT_X1:%.*]] = shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK:         [[EXT_Y1:%.*]] = shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK:         [[CALL1:%.*]] = call fast <4 x double> @__svml_pow4(<4 x double> [[EXT_X1]], <4 x double> [[EXT_Y1]])
;
; Concat partial results.
; CHECK:         [[CONCAT:%.*]] = shufflevector <4 x double> [[CALL0]], <4 x double> [[CALL1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK:         store <8 x double> [[CONCAT]], ptr
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %arrayidx.exp = getelementptr inbounds double, ptr %exponents, i64 %iv
  %exp = load double, ptr %arrayidx.exp, align 8
  %call = tail call fast double @pow(double %conv, double %exp)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = { nounwind readnone }

declare double @pow(double, double) #0

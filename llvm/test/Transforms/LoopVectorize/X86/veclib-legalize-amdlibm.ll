; RUN: opt -vector-library=AMDLIBM -passes=inject-tli-mappings,loop-vectorize \
; RUN:   -force-vector-width=8 -force-vector-interleave=1 -mattr=avx \
; RUN:   -legalize-vector-library-calls -S < %s | FileCheck %s

; Verify that vector library call legalization works with AMDLIBM.
; At VF=8 on AVX (256-bit), <8 x double> = 512 bits exceeds the register
; width. Each call is split into 2 x VF=4 calls using amd_vrd4_* variants.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @amdlibm_sin_f64(ptr nocapture %varray) {
; CHECK-LABEL: @amdlibm_sin_f64(
; CHECK:       vector.body:
; CHECK:         [[WIDE:%.*]] = sitofp <8 x i32> {{.*}} to <8 x double>
; CHECK:         [[EXT0:%.*]] = shufflevector <8 x double> [[WIDE]], <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK:         [[CALL0:%.*]] = call <4 x double> @amd_vrd4_sin(<4 x double> [[EXT0]])
; CHECK:         [[EXT1:%.*]] = shufflevector <8 x double> [[WIDE]], <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK:         [[CALL1:%.*]] = call <4 x double> @amd_vrd4_sin(<4 x double> [[EXT1]])
; CHECK:         [[CONCAT:%.*]] = shufflevector <4 x double> [[CALL0]], <4 x double> [[CALL1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK:         store <8 x double> [[CONCAT]], ptr
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @sin(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @amdlibm_cos_f64(ptr nocapture %varray) {
; CHECK-LABEL: @amdlibm_cos_f64(
; CHECK:       vector.body:
; CHECK:         [[WIDE2:%.*]] = sitofp <8 x i32> {{.*}} to <8 x double>
; CHECK:         [[EXT2_0:%.*]] = shufflevector <8 x double> [[WIDE2]], <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK:         [[CCALL0:%.*]] = call <4 x double> @amd_vrd4_cos(<4 x double> [[EXT2_0]])
; CHECK:         [[EXT2_1:%.*]] = shufflevector <8 x double> [[WIDE2]], <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK:         [[CCALL1:%.*]] = call <4 x double> @amd_vrd4_cos(<4 x double> [[EXT2_1]])
; CHECK:         [[CCONCAT:%.*]] = shufflevector <4 x double> [[CCALL0]], <4 x double> [[CCALL1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK:         store <8 x double> [[CCONCAT]], ptr
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @cos(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @amdlibm_exp_f64(ptr nocapture %varray) {
; CHECK-LABEL: @amdlibm_exp_f64(
; CHECK:       vector.body:
; CHECK:         [[WIDE3:%.*]] = sitofp <8 x i32> {{.*}} to <8 x double>
; CHECK:         [[EXT3_0:%.*]] = shufflevector <8 x double> [[WIDE3]], <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK:         [[ECALL0:%.*]] = call <4 x double> @amd_vrd4_exp(<4 x double> [[EXT3_0]])
; CHECK:         [[EXT3_1:%.*]] = shufflevector <8 x double> [[WIDE3]], <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK:         [[ECALL1:%.*]] = call <4 x double> @amd_vrd4_exp(<4 x double> [[EXT3_1]])
; CHECK:         [[ECONCAT:%.*]] = shufflevector <4 x double> [[ECALL0]], <4 x double> [[ECALL1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK:         store <8 x double> [[ECONCAT]], ptr
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @exp(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = { nounwind readnone }

declare double @sin(double) #0
declare double @cos(double) #0
declare double @exp(double) #0

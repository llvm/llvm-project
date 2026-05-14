; RUN: opt -vector-library=AMDLIBM -passes=inject-tli-mappings,loop-vectorize \
; RUN:   -force-vector-width=16 -force-vector-interleave=1 -mattr=avx \
; RUN:   -legalize-vector-library-calls -S < %s | FileCheck %s

; Verify that vector library call legalization works with AMDLIBM for single-
; precision (float) functions. At VF=16 on AVX (256-bit), <16 x float> = 512
; bits exceeds the register width. Each call is split into 2 x VF=8 calls
; using amd_vrs8_* variants.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @amdlibm_sinf(ptr nocapture %varray) {
; CHECK-LABEL: @amdlibm_sinf(
; CHECK:       vector.body:
; CHECK:         [[WIDE:%.*]] = sitofp <16 x i32> {{.*}} to <16 x float>
; CHECK:         [[EXT0:%.*]] = shufflevector <16 x float> [[WIDE]], <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK:         [[CALL0:%.*]] = call <8 x float> @amd_vrs8_sinf(<8 x float> [[EXT0]])
; CHECK:         [[EXT1:%.*]] = shufflevector <16 x float> [[WIDE]], <16 x float> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK:         [[CALL1:%.*]] = call <8 x float> @amd_vrs8_sinf(<8 x float> [[EXT1]])
; CHECK:         [[CONCAT:%.*]] = shufflevector <8 x float> [[CALL0]], <8 x float> [[CALL1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK:         store <16 x float> [[CONCAT]], ptr
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @sinf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @amdlibm_expf(ptr nocapture %varray) {
; CHECK-LABEL: @amdlibm_expf(
; CHECK:       vector.body:
; CHECK:         [[WIDE2:%.*]] = sitofp <16 x i32> {{.*}} to <16 x float>
; CHECK:         [[EXT2_0:%.*]] = shufflevector <16 x float> [[WIDE2]], <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK:         [[ECALL0:%.*]] = call <8 x float> @amd_vrs8_expf(<8 x float> [[EXT2_0]])
; CHECK:         [[EXT2_1:%.*]] = shufflevector <16 x float> [[WIDE2]], <16 x float> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK:         [[ECALL1:%.*]] = call <8 x float> @amd_vrs8_expf(<8 x float> [[EXT2_1]])
; CHECK:         [[ECONCAT:%.*]] = shufflevector <8 x float> [[ECALL0]], <8 x float> [[ECALL1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK:         store <16 x float> [[ECONCAT]], ptr
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @expf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = { nounwind readnone }

declare float @sinf(float) #0
declare float @expf(float) #0

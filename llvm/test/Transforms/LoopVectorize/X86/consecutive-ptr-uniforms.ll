; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -instcombine -S -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: PR31671
;
; Check a pointer in which one of its uses is consecutive-like and another of
; its uses is non-consecutive-like. In the test case below, %tmp3 is the
; pointer operand of an interleaved load, making it consecutive-like. However,
; it is also the pointer operand of a non-interleaved store that will become a
; scatter operation. %tmp3 (and the induction variable) should not be marked
; uniform-after-vectorization.
;
; CHECK:     LV: Found uniform instruction: %tmp0 = getelementptr inbounds %data, %data* %d, i64 0, i32 3, i64 %i
; CHECK-NOT: LV: Found uniform instruction: %tmp3 = getelementptr inbounds %data, %data* %d, i64 0, i32 0, i64 %i
; CHECK-NOT: LV: Found uniform instruction: %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
; CHECK-NOT: LV: Found uniform instruction: %i.next = add nuw nsw i64 %i, 5
; CHECK:     vector.body:
; CHECK:       %index = phi i64
; CHECK:       %vec.ind = phi <16 x i64>
; CHECK:       %[[T0:.+]] = mul i64 %index, 5
; CHECK:       %[[T1:.+]] = getelementptr inbounds %data, %data* %d, i64 0, i32 3, i64 %[[T0]]
; CHECK:       %[[T2:.+]] = bitcast float* %[[T1]] to <80 x float>*
; CHECK:       load <80 x float>, <80 x float>* %[[T2]], align 4
; CHECK:       %[[T3:.+]] = getelementptr inbounds %data, %data* %d, i64 0, i32 0, i64 %[[T0]]
; CHECK:       %[[T4:.+]] = bitcast float* %[[T3]] to <80 x float>*
; CHECK:       load <80 x float>, <80 x float>* %[[T4]], align 4
; CHECK:       %VectorGep = getelementptr inbounds %data, %data* %d, i64 0, i32 0, <16 x i64> %vec.ind
; CHECK:       call void @llvm.masked.scatter.v16f32({{.*}}, <16 x float*> %VectorGep, {{.*}})
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body

%data = type { [32000 x float], [3 x i32], [4 x i8], [32000 x float] }

define void @PR31671(float %x, %data* %d) #0 {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds %data, %data* %d, i64 0, i32 3, i64 %i
  %tmp1 = load float, float* %tmp0, align 4
  %tmp2 = fmul float %x, %tmp1
  %tmp3 = getelementptr inbounds %data, %data* %d, i64 0, i32 0, i64 %i
  %tmp4 = load float, float* %tmp3, align 4
  %tmp5 = fadd float %tmp4, %tmp2
  store float %tmp5, float* %tmp3, align 4
  %i.next = add nuw nsw i64 %i, 5
  %cond = icmp slt i64 %i.next, 32000
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

attributes #0 = { "target-cpu"="knl" }

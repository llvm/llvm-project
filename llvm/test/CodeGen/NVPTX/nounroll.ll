; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; Compiled from the following CUDA code:
;
;   #pragma nounroll
;   for (int i = 0; i < 2; ++i)
;     output[i] = input[i];
define void @nounroll(float* %input, float* %output) {
; CHECK-LABEL: .visible .func nounroll(
entry:
  br label %for.body

for.body:
; CHECK: .pragma "nounroll"
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = sext i32 %i.06 to i64
  %arrayidx = getelementptr inbounds float, float* %input, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
; CHECK: ld.f32
  %arrayidx2 = getelementptr inbounds float, float* %output, i64 %idxprom
  store float %0, float* %arrayidx2, align 4
; CHECK: st.f32
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 2
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0
; CHECK-NOT: ld.f32
; CHECK-NOT: st.f32

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}

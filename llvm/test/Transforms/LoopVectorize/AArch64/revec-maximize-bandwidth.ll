; REQUIRES: asserts
; RUN: opt -disable-output -passes=loop-vectorize -vectorize-vector-loops \
; RUN:     -mtriple=aarch64 -mattr=+sve \
; RUN:     -vectorizer-maximize-bandwidth=false -debug-only=loop-vectorize \
; RUN:     < %s 2>&1 | FileCheck %s --check-prefix=NOMAXBW
; RUN: opt -disable-output -passes=loop-vectorize -vectorize-vector-loops \
; RUN:     -mtriple=aarch64 -mattr=+sve \
; RUN:     -vectorizer-maximize-bandwidth=true -debug-only=loop-vectorize \
; RUN:     < %s 2>&1 | FileCheck %s --check-prefix=MAXBW

; The widest value fills a 128-bit NEON register. Without maximizing bandwidth,
; it limits the scalable VF to vscale x 1. The smaller 64-bit value allows
; vscale x 2 when maximizing bandwidth.

; NOMAXBW-LABEL: LV: Checking a loop in 'max_bandwidth'
; NOMAXBW: LV: The Smallest and Widest types: 64 / 128 bits.
; NOMAXBW: LV: The Widest register safe to use is: 128 bits.
; NOMAXBW-NEXT: LV: The Widest register safe to use is: vscale x 128 bits.
; NOMAXBW-NEXT: LV: Found feasible scalable VF = vscale x 1

; MAXBW-LABEL: LV: Checking a loop in 'max_bandwidth'
; MAXBW: LV: The Smallest and Widest types: 64 / 128 bits.
; MAXBW: LV: The Widest register safe to use is: 128 bits.
; MAXBW-NEXT: LV: The Widest register safe to use is: vscale x 128 bits.
; MAXBW-NEXT: LV: Found feasible scalable VF = vscale x 2

define void @max_bandwidth(ptr noalias %dst16, ptr noalias %src16,
                           ptr noalias %dst8, ptr noalias %src8, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %src16.gep = getelementptr inbounds <16 x i8>, ptr %src16, i64 %iv
  %dst16.gep = getelementptr inbounds <16 x i8>, ptr %dst16, i64 %iv
  %v16 = load <16 x i8>, ptr %src16.gep, align 16
  %add16 = add <16 x i8> %v16, splat (i8 1)
  store <16 x i8> %add16, ptr %dst16.gep, align 16
  %src8.gep = getelementptr inbounds <8 x i8>, ptr %src8, i64 %iv
  %dst8.gep = getelementptr inbounds <8 x i8>, ptr %dst8, i64 %iv
  %v8 = load <8 x i8>, ptr %src8.gep, align 8
  %add8 = add <8 x i8> %v8, splat (i8 1)
  store <8 x i8> %add8, ptr %dst8.gep, align 8
  %iv.next = add nuw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

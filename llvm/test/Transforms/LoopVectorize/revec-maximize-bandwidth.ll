; REQUIRES: asserts
; RUN: opt -disable-output -passes=loop-vectorize -vectorize-vector-loops \
; RUN:     -force-target-supports-scalable-vectors -scalable-vectorization=on \
; RUN:     -vectorizer-maximize-bandwidth=false -debug-only=loop-vectorize \
; RUN:     < %s 2>&1 | FileCheck %s --check-prefix=NOMAXBW
; RUN: opt -disable-output -passes=loop-vectorize -vectorize-vector-loops \
; RUN:     -force-target-supports-scalable-vectors -scalable-vectorization=on \
; RUN:     -vectorizer-maximize-bandwidth=true -debug-only=loop-vectorize \
; RUN:     < %s 2>&1 | FileCheck %s --check-prefix=MAXBW

; Without maximizing bandwidth, the widest type limits the scalable VF to
; vscale x 1. When maximizing bandwidth, the smallest type allows vscale x 2.

; NOMAXBW-LABEL: LV: Checking a loop in 'max_bandwidth'
; NOMAXBW: LV: The Smallest and Widest types: 16 / 32 bits.
; NOMAXBW: LV: Found feasible scalable VF = vscale x 1

; MAXBW-LABEL: LV: Checking a loop in 'max_bandwidth'
; MAXBW: LV: The Smallest and Widest types: 16 / 32 bits.
; MAXBW: LV: Found feasible scalable VF = vscale x 2

define void @max_bandwidth(ptr noalias %dst4, ptr noalias %src4,
                           ptr noalias %dst2, ptr noalias %src2, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %src4.gep = getelementptr inbounds <4 x i8>, ptr %src4, i64 %iv
  %dst4.gep = getelementptr inbounds <4 x i8>, ptr %dst4, i64 %iv
  %v4 = load <4 x i8>, ptr %src4.gep, align 4
  %add4 = add <4 x i8> %v4, splat (i8 1)
  store <4 x i8> %add4, ptr %dst4.gep, align 4
  %src2.gep = getelementptr inbounds <2 x i8>, ptr %src2, i64 %iv
  %dst2.gep = getelementptr inbounds <2 x i8>, ptr %dst2, i64 %iv
  %v2 = load <2 x i8>, ptr %src2.gep, align 2
  %add2 = add <2 x i8> %v2, splat (i8 1)
  store <2 x i8> %add2, ptr %dst2.gep, align 2
  %iv.next = add nuw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

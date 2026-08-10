; RUN: opt -loop-reduce -S < %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; LSR used not to be able to generate a float* induction variable in
; these cases due to scalar evolution not propagating nsw from an
; instruction to the SCEV, preventing distributing sext into the
; corresponding addrec.

; Test this pattern:
;
;   for (int i = 0; i < numIterations; ++i)
;     sum += ptr[i + offset];
;
define float @testadd(ptr %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @testadd
; CHECK: sext i32 %offset to i64
; CHECK: loop:
; CHECK-DAG: phi ptr
; CHECK-DAG: phi i32
; CHECK-NOT: sext

entry:
  br label %loop

loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %sum = phi float [ %nextsum, %loop ], [ 0.000000e+00, %entry ]
  %index32 = add nuw nsw i32 %i, %offset
  %index64 = sext i32 %index32 to i64
  %ptr = getelementptr inbounds float, ptr %input, i64 %index64
  %addend = load float, ptr %ptr, align 4
  %nextsum = fadd float %sum, %addend
  %nexti = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret float %nextsum
}

; Test this pattern:
;
;   for (int i = 0; i < numIterations; ++i)
;     sum += ptr[i - offset];
;
define float @testsub(ptr %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @testsub
; CHECK: sext i32 %offset to i64
; CHECK: loop:
; CHECK-DAG: phi ptr
; CHECK-DAG: phi i32
; CHECK-NOT: sext

entry:
  br label %loop

loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %sum = phi float [ %nextsum, %loop ], [ 0.000000e+00, %entry ]
  %index32 = sub nuw nsw i32 %i, %offset
  %index64 = sext i32 %index32 to i64
  %ptr = getelementptr inbounds float, ptr %input, i64 %index64
  %addend = load float, ptr %ptr, align 4
  %nextsum = fadd float %sum, %addend
  %nexti = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret float %nextsum
}

; Test this pattern:
;
;   for (int i = 0; i < numIterations; ++i)
;     sum += ptr[i * stride];
;
define float @testmul(ptr %input, i32 %stride, i32 %numIterations) {
; CHECK-LABEL: @testmul
; CHECK: sext i32 %stride to i64
; CHECK: loop:
; CHECK-DAG: phi ptr
; CHECK-DAG: phi i32
; CHECK-NOT: sext

entry:
  br label %loop

loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %sum = phi float [ %nextsum, %loop ], [ 0.000000e+00, %entry ]
  %index32 = mul nuw nsw i32 %i, %stride
  %index64 = sext i32 %index32 to i64
  %ptr = getelementptr inbounds float, ptr %input, i64 %index64
  %addend = load float, ptr %ptr, align 4
  %nextsum = fadd float %sum, %addend
  %nexti = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret float %nextsum
}

; Test this pattern:
;
;   for (int i = 0; i < numIterations; ++i)
;     sum += ptr[3 * (i << 7)];
;
; The multiplication by 3 is to make the address calculation expensive
; enough to force the introduction of a pointer induction variable.
define float @testshl(ptr %input, i32 %numIterations) {
; CHECK-LABEL: @testshl
; CHECK: loop:
; CHECK-DAG: phi ptr
; CHECK-DAG: phi i32
; CHECK-NOT: sext

entry:
  br label %loop

loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %sum = phi float [ %nextsum, %loop ], [ 0.000000e+00, %entry ]
  %index32 = shl nuw nsw i32 %i, 7
  %index32mul = mul nuw nsw i32 %index32, 3
  %index64 = sext i32 %index32mul to i64
  %ptr = getelementptr inbounds float, ptr %input, i64 %index64
  %addend = load float, ptr %ptr, align 4
  %nextsum = fadd float %sum, %addend
  %nexti = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret float %nextsum
}

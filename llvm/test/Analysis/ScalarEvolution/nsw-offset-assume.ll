; RUN: opt < %s -S -analyze -scalar-evolution | FileCheck %s

; ScalarEvolution should be able to fold away the sign-extensions
; on this loop with a primary induction variable incremented with
; a nsw add of 2 (this test is derived from the nsw-offset.ll test, but uses an
; assume instead of a preheader conditional branch to guard the loop).

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @foo(i32 %no, double* nocapture %d, double* nocapture %q) nounwind {
entry:
  %n = and i32 %no, 4294967294
  %0 = icmp sgt i32 %n, 0                         ; <i1> [#uses=1]
  tail call void @llvm.assume(i1 %0)
  br label %bb.nph

bb.nph:                                           ; preds = %entry
  br label %bb

bb:                                               ; preds = %bb.nph, %bb1
  %i.01 = phi i32 [ %16, %bb1 ], [ 0, %bb.nph ]   ; <i32> [#uses=5]

; CHECK: %1 = sext i32 %i.01 to i64
; CHECK: -->  {0,+,2}<nuw><nsw><%bb>
  %1 = sext i32 %i.01 to i64                      ; <i64> [#uses=1]

; CHECK: %2 = getelementptr inbounds double, double* %d, i64 %1
; CHECK: -->  {%d,+,16}<nsw><%bb>
  %2 = getelementptr inbounds double, double* %d, i64 %1  ; <double*> [#uses=1]

  %3 = load double, double* %2, align 8                   ; <double> [#uses=1]
  %4 = sext i32 %i.01 to i64                      ; <i64> [#uses=1]
  %5 = getelementptr inbounds double, double* %q, i64 %4  ; <double*> [#uses=1]
  %6 = load double, double* %5, align 8                   ; <double> [#uses=1]
  %7 = or i32 %i.01, 1                            ; <i32> [#uses=1]

; CHECK: %8 = sext i32 %7 to i64
; CHECK: -->  {1,+,2}<nuw><nsw><%bb>
  %8 = sext i32 %7 to i64                         ; <i64> [#uses=1]

; CHECK: %9 = getelementptr inbounds double, double* %q, i64 %8
; CHECK: {(8 + %q)<nsw>,+,16}<nsw><%bb>
  %9 = getelementptr inbounds double, double* %q, i64 %8  ; <double*> [#uses=1]

; Artificially repeat the above three instructions, this time using
; add nsw instead of or.
  %t7 = add nsw i32 %i.01, 1                            ; <i32> [#uses=1]

; CHECK: %t8 = sext i32 %t7 to i64
; CHECK: -->  {1,+,2}<nuw><nsw><%bb>
  %t8 = sext i32 %t7 to i64                         ; <i64> [#uses=1]

; CHECK: %t9 = getelementptr inbounds double, double* %q, i64 %t8
; CHECK: {(8 + %q)<nsw>,+,16}<nsw><%bb>
  %t9 = getelementptr inbounds double, double* %q, i64 %t8  ; <double*> [#uses=1]

  %10 = load double, double* %9, align 8                  ; <double> [#uses=1]
  %11 = fadd double %6, %10                       ; <double> [#uses=1]
  %12 = fadd double %11, 3.200000e+00             ; <double> [#uses=1]
  %13 = fmul double %3, %12                       ; <double> [#uses=1]
  %14 = sext i32 %i.01 to i64                     ; <i64> [#uses=1]
  %15 = getelementptr inbounds double, double* %d, i64 %14 ; <double*> [#uses=1]
  store double %13, double* %15, align 8
  %16 = add nsw i32 %i.01, 2                      ; <i32> [#uses=2]
  br label %bb1

bb1:                                              ; preds = %bb
  %17 = icmp slt i32 %16, %n                      ; <i1> [#uses=1]
  br i1 %17, label %bb, label %bb1.return_crit_edge

bb1.return_crit_edge:                             ; preds = %bb1
  br label %return

return:                                           ; preds = %bb1.return_crit_edge, %entry
  ret void
}

declare void @llvm.assume(i1) nounwind

; Note: Without the preheader assume, there is an 'smax' in the
; backedge-taken count expression:
; CHECK: Loop %bb: backedge-taken count is ((-1 + (2 * (%no /u 2))<nuw>) /u 2)
; CHECK: Loop %bb: max backedge-taken count is 1073741822

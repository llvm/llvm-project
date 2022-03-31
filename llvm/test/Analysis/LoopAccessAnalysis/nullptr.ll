; RUN: opt -passes='print-access-info' -disable-output  < %s 2>&1 | FileCheck %s

; Test that the loop accesses are proven safe in this case.
; The analyzer uses to be confused by the "diamond" because getUnderlyingObjects
; is saying that the two pointers can both points to null. The loop analyzer
; needs to ignore null in the results returned by getUnderlyingObjects.

; CHECK: Memory dependences are safe with run-time checks


; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Function Attrs: ssp uwtable
define void @foo(i1 %cond, i32* %ptr1, i32* %ptr2)  {
  br i1 %cond, label %.preheader, label %diamond

diamond:  ; preds = %.noexc.i.i
  br label %.preheader

.preheader:                                     ; preds = %diamond, %0
  %ptr1_or_null = phi i32* [ null, %0 ], [ %ptr1, %diamond ]
  %ptr2_or_null = phi i32* [ null, %0 ], [ %ptr2, %diamond ]
  br label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph, %.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 10, %.preheader ]
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %tmp4 = getelementptr inbounds i32, i32* %ptr2_or_null, i64 %indvars.iv.next
  %tmp5 = load i32, i32* %tmp4, align 4
  %tmp6 = getelementptr inbounds i32, i32* %ptr1_or_null, i64 %indvars.iv.next
  store i32 undef, i32* %tmp6, align 4
  br i1 false, label %.lr.ph, label %.end

.end:
  ret void
}

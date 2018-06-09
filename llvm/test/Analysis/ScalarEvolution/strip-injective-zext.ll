; RUN: opt -analyze -scalar-evolution < %s | FileCheck %s

; The initial SCEV for the backedge count is
;   (zext i2 {(trunc i32 (1 + %a1) to i2),+,1}<%b2> to i32).
; In howFarToZero, this was further converted to an add-rec, the complexity
; of which defeated the calculation of the backedge taken count.
; Since such zero-extensions preserve the values being extended, strip
; them in howFarToZero to simplify the input SCEV.

; Check that the backedge taken count was actually computed:
; CHECK: Determining loop execution counts for: @f0
; CHECK-NEXT: Loop %b2: backedge-taken count is (-1 * (trunc i32 (1 + %a1) to i2))

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"

define i32 @f0(i32 %a0, i32 %a1, i32* nocapture %a2) #0 {
b0:
  %v0 = and i32 %a1, 3
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %b4, label %b1

b1:                                               ; preds = %b0
  %v2 = shl i32 %a0, 7
  %v3 = add i32 %v2, -128
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v4 = phi i32 [ %a1, %b1 ], [ %v9, %b2 ]
  %v5 = phi i32* [ %a2, %b1 ], [ %v8, %b2 ]
  %v6 = getelementptr inbounds i32, i32* %v5, i32 0
  store i32 %v3, i32* %v6, align 4
  %v8 = getelementptr inbounds i32, i32* %v5, i32 1
  %v9 = add nsw i32 %v4, 1
  %v10 = and i32 %v9, 3
  %v11 = icmp eq i32 %v10, 0
  br i1 %v11, label %b3, label %b2

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  ret i32 0
}

attributes #0 = { norecurse nounwind }

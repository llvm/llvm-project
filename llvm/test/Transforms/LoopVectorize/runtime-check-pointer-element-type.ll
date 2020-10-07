; RUN: opt -loop-vectorize -force-vector-width=2 %s -S -debug 2>&1 | FileCheck %s
; RUN: opt -passes='loop-vectorize' -force-vector-width=2 %s -S -debug 2>&1 | FileCheck %s

; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; FIXME
; Test case for PR47751. Make sure the runtime check includes a required
; addition of the size of the element type (a pointer) for the end bound.

define void @test(i64 %arg, i32 %arg1, i8** %base) {
; CHECK:      LAA: Adding RT check for range:
; CHECK-NEXT:  Start: ((8 * (zext i32 (-1 + %arg1)<nsw> to i64))<nuw><nsw> + (8 * (1 smin %arg)) + (-8 * %arg) + %base)
; CHECK-SAME:  End: ((8 * (zext i32 (-1 + %arg1)<nsw> to i64))<nuw><nsw> + %base)
; CHECK-NEXT: LAA: Adding RT check for range:
; CHECK-NEXT:  Start: ((8 * (1 smin %arg)) + %base)
; CHECK-SAME:  End: ((8 * %arg) + %base)<nsw>

; CHECK: vector.body

entry:
  br label %loop

loop:
  %iv.1 = phi i64 [ %arg, %entry ], [ %iv.1.next, %loop ]
  %iv.2 = phi i32 [ %arg1, %entry ], [ %iv.2.next, %loop ]
  %iv.2.next = add nsw i32 %iv.2, -1
  %iv.2.ext = zext i32 %iv.2.next to i64
  %idx.1 = getelementptr inbounds i8*, i8** %base, i64 %iv.2.ext
  %v.1 = load i8*, i8** %idx.1, align 8
  %idx.2 = getelementptr inbounds i8*, i8** %base, i64 %iv.1
  %v.2 = load i8*, i8** %idx.2, align 8
  store i8* %v.2, i8** %idx.1, align 8
  store i8* %v.1, i8** %idx.2, align 8
  %tmp11 = icmp sgt i64 %iv.1, 1
  %iv.1.next = add nsw i64 %iv.1, -1
  br i1 %tmp11, label %loop, label %exit

exit:                                             ; preds = %bb3
  ret void
}

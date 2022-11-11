; Compiler should not be broken with a dead user.
; RUN: opt -passes=sink -S < %s | FileCheck %s

; CHECK-LABEL: @test(
; CHECK:       bb.0:
; CHECK-NEXT:    %conv = sext i16 %p1 to i32
; CHECK-NEXT:    br i1 undef, label %bb.1, label %bb.3

; CHECK:       bb.1:                                             ; preds = %bb.0
; CHECK-NEXT:    br label %bb.2

; CHECK:       bb.2:                                             ; preds = %bb.2, %bb.1
; CHECK-NEXT:    %and.2 = and i32 undef, %conv
; CHECK-NEXT:    br label %bb.2

; CHECK:       bb.3:                                             ; preds = %bb.3, %bb.0
; CHECK-NEXT:    %and.3 = and i32 undef, %conv
; CHECK-NEXT:    br label %bb.3

; CHECK:       dead:                                             ; preds = %dead
; CHECK-NEXT:    %and.dead = and i32 undef, %conv
; CHECK-NEXT:    br label %dead
define void @test(i16 %p1) {
bb.0:
  %conv = sext i16 %p1 to i32
  br i1 undef, label %bb.1, label %bb.3

bb.1:                                             ; preds = %bb.0
  br label %bb.2

bb.2:                                             ; preds = %bb.2, %bb.1
  %and.2 = and i32 undef, %conv
  br label %bb.2

bb.3:                                             ; preds = %bb.3, %bb.0
  %and.3 = and i32 undef, %conv
  br label %bb.3

dead:                                             ; preds = %dead
  %and.dead = and i32 undef, %conv
  br label %dead
}

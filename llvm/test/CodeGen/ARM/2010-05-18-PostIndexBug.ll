; RUN: llc < %s -mtriple=armv7-apple-darwin   | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s -check-prefix=THUMB
; rdar://7998649

%struct.foo = type { i64, i64 }

define zeroext i8 @t(%struct.foo* %this) noreturn optsize {
entry:
; ARM:       t:
; ARM:       str r2, [r1], r0

; THUMB:     t:
; THUMB-NOT: str r0, [r1], r0
; THUMB:     str r1, [r0]
  %0 = getelementptr inbounds %struct.foo* %this, i32 0, i32 1 ; <i64*> [#uses=1]
  store i32 0, i32* inttoptr (i32 8 to i32*), align 8
  br i1 undef, label %bb.nph96, label %bb3

bb3:                                              ; preds = %entry
  %1 = load i64* %0, align 4                      ; <i64> [#uses=0]
  unreachable

bb.nph96:                                         ; preds = %entry
  unreachable
}

; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck --check-prefix=NO-THREADING %s
; Checks that we do not thread the control flow through the loop header bb1 as
; that will introduce a non-canonical loop

; NO-THREADING-LABEL: define void @__start
; NO-THREADING: bb3:
; NO-THREADING-NEXT: br i1 %cond, label %bb1, label %bb5

; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 --keep-loops="false" -S | FileCheck --check-prefix=THREADING %s
; Checks that we thread the control flow through the loop header bb1 since we
; do not request --keep-loops

; THREADING-LABEL: define void @__start
; THREADING: bb3:
; THREADING-NEXT: br i1 %cond, label %bb4, label %bb5

define void @__start(i1 %cond) {
entry:
  br label %bb1

bb1:                                            ; preds = %bb3, %entry
  br i1 %cond, label %bb4, label %bb2

bb2:                                            ; preds = %bb1
  %_0_ = add i16 0, 0
  br label %bb3

bb3:                                            ; preds = %bb4, %bb2
  br i1 %cond, label %bb1, label %bb5

bb4:                                            ; preds = %bb1
  %_1_ = add i32 0, 1
  br label %bb3

bb5:                                            ; preds = %bb3
  ret void
}

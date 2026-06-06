; RUN: opt < %s -passes='loop(loop-deletion),print<domtree>' -verify-dom-info 2>&1 | FileCheck -check-prefix=DT %s

; CHECK: for.body
; CHECK-NOT: for.cond1

; Verify only the important parts of the DomTree.
; DT: [1] %entry
; DT:   [2] %for.cond
; DT:     [3] %lbl63A679E5
; DT:     [3] %for.cond9
; DT:     [3] %lbl64774A9B
; DT:     [3] %for.body
; DT:       [4] %for.cond3.loopexit

define i32 @fn1() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %entry
  br i1 true, label %lbl63A679E5, label %for.body

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.cond1, %for.body
  br i1 false, label %for.cond1, label %for.cond3.loopexit

for.cond3.loopexit:                               ; preds = %for.cond1
  br label %for.cond3

for.cond3:                                        ; preds = %for.cond9, %for.cond3.loopexit
  br i1 false, label %for.body4, label %for.cond17

for.body4:                                        ; preds = %for.cond3
  br label %for.cond5

for.cond5:                                        ; preds = %lbl63A679E5, %for.body4
  br label %for.cond9

lbl63A679E5:                                      ; preds = %for.cond
  br label %for.cond5

for.cond9:                                        ; preds = %for.end14.split, %for.cond5
  br i1 false, label %for.cond3, label %lbl64774A9B

lbl64774A9B:                                      ; preds = %for.cond17, %for.cond9
  br label %for.end14.split

for.end14.split:                                  ; preds = %lbl64774A9B
  br label %for.cond9

for.cond17:                                       ; preds = %for.cond3
  br label %lbl64774A9B
}

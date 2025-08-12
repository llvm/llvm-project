; RUN: opt -passes=loop-rotate -verify-memoryssa -S %s | FileCheck %s
; REQUIRES: asserts

; CHECK-LABEL: @func_35(i1 %arg)
define void @func_35(i1 %arg) {
entry:
  br i1 %arg, label %for.cond1704.preheader, label %return

for.cond1704.preheader:                           ; preds = %entry
  br label %for.cond1704

for.cond1704:                                     ; preds = %for.cond1704.preheader, %for.body1707
  br i1 false, label %for.body1707, label %return.loopexit

for.body1707:                                     ; preds = %for.cond1704
  store i32 1712, ptr undef, align 1
  br label %for.cond1704

for.body1102:                                     ; preds = %for.body1102
  br i1 %arg, label %for.body1102, label %return

return.loopexit:                                  ; preds = %for.cond1704
  br label %return

return:                                           ; preds = %return.loopexit, %for.body1102, %entry
  ret void
}

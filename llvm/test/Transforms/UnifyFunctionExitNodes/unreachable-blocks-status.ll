; RUN: opt -mergereturn -enable-new-pm=0 -S < %s | FileCheck %s
; RUN: opt -passes='break-crit-edges,lowerswitch,mergereturn' -S < %s | FileCheck %s

; The pass did previously not report the correct Modified status in the case
; where a function had at most one return block, and an unified unreachable
; block was created. This was caught by the pass return status check that is
; hidden under EXPENSIVE_CHECKS.

; CHECK: for.foo.body2:
; CHECK-NEXT: br label %UnifiedUnreachableBlock

; CHECK: for.foo.end:
; CHECK-NEXT: br label %UnifiedUnreachableBlock

; CHECK: UnifiedUnreachableBlock:
; CHECK-NEXT: unreachable

define i32 @foo() {
entry:
  br label %for.foo.cond

for.foo.cond:                                         ; preds = %entry
  br i1 false, label %for.foo.body, label %for.foo.end3

for.foo.body:                                         ; preds = %for.foo.cond
  br label %for.foo.cond1

for.foo.cond1:                                        ; preds = %for.foo.body
  br i1 false, label %for.foo.body2, label %for.foo.end

for.foo.body2:                                        ; preds = %for.foo.cond1
  unreachable

for.foo.end:                                          ; preds = %for.foo.cond1
  unreachable

for.foo.end3:                                         ; preds = %for.foo.cond
  ret i32 undef
}

; CHECK: for.bar.body2:
; CHECK-NEXT: br label %UnifiedUnreachableBlock

; CHECK: for.bar.end:
; CHECK-NEXT: br label %UnifiedUnreachableBlock

; CHECK: UnifiedUnreachableBlock:
; CHECK-NEXT: unreachable

define void @bar() {
entry:
  br label %for.bar.cond

for.bar.cond:                                         ; preds = %entry
  br i1 false, label %for.bar.body, label %for.bar.end

for.bar.body:                                         ; preds = %for.bar.cond
  br label %for.bar.cond1

for.bar.cond1:                                        ; preds = %for.bar.body
  br i1 false, label %for.bar.body2, label %for.bar.end

for.bar.body2:                                        ; preds = %for.bar.cond1
  unreachable

for.bar.end:                                          ; preds = %for.bar.cond1
  unreachable
}

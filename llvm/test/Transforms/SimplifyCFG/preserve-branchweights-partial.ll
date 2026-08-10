; RUN: opt -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S -o - < %s | FileCheck %s

; This test case was written to trigger an incorrect assert statement in
; -simplifycfg.  Thus we don't actually want to check the output, just that
; -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 ran successfully.  Thus we only check that the function still
; exists, and that it still calls foo().
;
; NOTE: There are some obviously dead blocks and missing branch weight
;       metadata.  Both of these features were key to triggering the assert.
;       Additionally, the not-taken weight of the branch with a weight had to
;       be 0 to trigger the assert.

declare void @foo() nounwind uwtable

define void @func(i32 %A) nounwind uwtable {
; CHECK-LABEL: define void @func(
entry:
  %cmp11 = icmp eq i32 %A, 1
  br i1 %cmp11, label %if.then, label %if.else, !prof !0

if.then:
  call void @foo()
; CHECK: call void @foo()
  br label %if.else

if.else:
  %cmp17 = icmp eq i32 %A, 2
  br i1 %cmp17, label %if.then2, label %if.end

if.then2:
  br label %if.end

if.end:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 0}

; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; Checks that the SimplifyCFG pass won't duplicate a call to a function marked
; convergent.
;
; CHECK: call void @barrier
; CHECK-NOT: call void @barrier
define void @check(i1 %cond, ptr %out) {
entry:
  br i1 %cond, label %if.then, label %if.end

if.then:
  store i32 5, ptr %out
  br label %if.end

if.end:
  %x = phi i1 [ true, %entry ], [ false, %if.then ]
  call void @barrier()
  br i1 %x, label %cond.end, label %cond.false

cond.false:
  br label %cond.end

cond.end:
  ret void
}

declare void @barrier() convergent

; RUN: opt < %s -simplifycfg -S | FileCheck %s

; Checks that the SimplifyCFG pass won't duplicate a call to a function marked
; convergent.
;
; CHECK: call void @barrier
; CHECK-NOT: call void @barrier
define void @check(i1 %cond, i32* %out) {
entry:
  br i1 %cond, label %if.then, label %if.end

if.then:
  store i32 5, i32* %out
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

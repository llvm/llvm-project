; RUN: opt -passes=dead-branch-elim -S %s | FileCheck %s

; Loop control flow must never be touched: exit edges of counted loops and
; runtime-bound loops are "not provable" in early iterations of the fixed
; point but must end up ProvenReachable, otherwise loops become infinite.

; CHECK-LABEL: define void @counted_loop()
; CHECK: loop:
; CHECK: br i1 %cmp, label %loop, label %exit
; CHECK: exit:
define void @counted_loop() {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %cmp = icmp ult i32 %next, 100
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; CHECK-LABEL: define i32 @runtime_bound(i32 %n)
; CHECK: loop:
; CHECK: br i1 %cmp, label %body, label %exit
; CHECK: body:
; CHECK: exit:
define i32 @runtime_bound(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %body ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %body, label %exit

body:
  %sum.next = add nsw i32 %sum, %i
  %i.next = add nsw i32 %i, 1
  br label %loop

exit:
  ret i32 %sum
}

; Conditions that cannot be analyzed (argument-dependent, memory-based) must
; keep their branches.
; CHECK-LABEL: define void @arg_cond(i32 %n)
; CHECK: br i1 %cmp, label %then, label %done
; CHECK: then:
declare void @side_effect()
define void @arg_cond(i32 %n) {
entry:
  %cmp = icmp eq i32 %n, 42
  br i1 %cmp, label %then, label %done

then:
  call void @side_effect()
  br label %done

done:
  ret void
}

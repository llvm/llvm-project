; RUN: opt -S -mtriple=bpf -passes='bpf-func-table,simplifycfg' < %s | FileCheck %s

; No function of the type of the call has its address taken. A call of null is
; undefined behaviour: the exit of the loop that leads to it would go away and
; the kernel rejects loops that have no exit. It is a trap.

define i64 @count(ptr %f, i64 %n) {
; CHECK-LABEL: @count
; CHECK: loop:
; CHECK: br i1 %{{.*}}, label %[[EXIT:.*]], label %loop
; CHECK: [[EXIT]]:
; CHECK: call void @llvm.trap()
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  %r = call i64 %f(i64 %next)
  ret i64 %r
}

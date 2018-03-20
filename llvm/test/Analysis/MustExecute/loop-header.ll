; RUN: opt -analyze -print-mustexecute %s

; CHECK: Printing analysis 'Instructions which execute on loop entry' for function 'header_with_icf':
; CHECK: The following are guaranteed to execute (for the respective loops):
; CHECK:   %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]	(mustexec in: loop)
; CHECK:   %v = load i32, i32* %p	(mustexec in: loop)
; CHECK:   call void @maythrow_and_use(i32 %v)	(mustexec in: loop)
; CHECK-NOT: add
define i1 @header_with_icf(i32* noalias %p, i32 %high) {
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  %v = load i32, i32* %p
  call void @maythrow_and_use(i32 %v)
  %iv.next = add nsw nuw i32 %iv, 1
  %exit.test = icmp slt i32 %iv, %high
  br i1 %exit.test, label %exit, label %loop

exit:
  ret i1 false
}

; CHECK: Printing analysis 'Instructions which execute on loop entry' for function 'test':
; CHECK: The following are guaranteed to execute (for the respective loops):
; CHECK:   %iv = phi i32 [ 0, %entry ], [ %iv.next, %next ]	(mustexec in: loop)
; CHECK:   %v = load i32, i32* %p	(mustexec in: loop)
; CHECK:   br label %next	(mustexec in: loop)
define i1 @test(i32* noalias %p, i32 %high) {
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %next]
  %v = load i32, i32* %p
  br label %next
next:
  call void @maythrow_and_use(i32 %v)
  %iv.next = add nsw nuw i32 %iv, 1
  %exit.test = icmp slt i32 %iv, %high
  br i1 %exit.test, label %exit, label %loop

exit:
  ret i1 false
}

; CHECK: Printing analysis 'Instructions which execute on loop entry' for function 'nested':
; CHECK: The following are guaranteed to execute (for the respective loops):
; CHECK:   %iv = phi i32 [ 0, %entry ], [ %iv.next, %next ]	(mustexec in: loop)
; CHECK:   br label %inner_loop	(mustexec in: loop)
; FIXME: These three are also must execute for the outer loop.
; CHECK:   %v = load i32, i32* %p	(mustexec in: inner_loop)
; CHECK:   %inner.test = icmp eq i32 %v, 0	(mustexec in: inner_loop)
; CHECK:   br i1 %inner.test, label %inner_loop, label %next	(mustexec in: inner_loop)
define i1 @nested(i32* noalias %p, i32 %high) {
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %next]
  br label %inner_loop
  
inner_loop:
  %v = load i32, i32* %p
  %inner.test = icmp eq i32 %v, 0
  br i1 %inner.test, label %inner_loop, label %next

next:
  call void @maythrow_and_use(i32 %v)
  %iv.next = add nsw nuw i32 %iv, 1
  %exit.test = icmp slt i32 %iv, %high
  br i1 %exit.test, label %exit, label %loop

exit:
  ret i1 false
}


declare void @maythrow_and_use(i32)

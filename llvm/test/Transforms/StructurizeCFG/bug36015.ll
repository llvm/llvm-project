; RUN: opt -S -structurizecfg %s | FileCheck %s

; r321751 introduced a bug where control flow branching from if to exit was
; not handled properly and instead ended up in an infinite loop.
define void @bug36015(i32 %cmp0, i32 %count) {
entry:
  br label %loop.outer

loop.outer:
  %ctr.loop.outer = phi i32 [ 0, %entry ], [ %ctr.else, %else ]
  call void @foo(i32 0)
  br label %loop.inner

loop.inner:
  %ctr.loop.inner = phi i32 [ %ctr.loop.outer, %loop.outer ], [ %ctr.if, %if ]
  call void @foo(i32 1)
  %cond.inner = icmp eq i32 %cmp0, %ctr.loop.inner
  br i1 %cond.inner, label %if, label %else

; CHECK: if:
; CHECK:   %0 = xor i1 %cond.if, true
; CHECK:   br label %Flow
if:
  %ctr.if = add i32 %ctr.loop.inner, 1
  call void @foo(i32 2)
  %cond.if = icmp slt i32 %ctr.if, %count
  br i1 %cond.if, label %loop.inner, label %exit

; CHECK: Flow:
; CHECK:   %2 = phi i1 [ %0, %if ], [ true, %loop.inner ]
; CHECK:   %3 = phi i1 [ false, %if ], [ true, %loop.inner ]
; CHECK:   br i1 %2, label %Flow1, label %loop.inner

; CHECK: Flow1:
; CHECK:   br i1 %3, label %else, label %Flow2

; CHECK: else:
; CHECK:   br label %Flow2
else:
  %ctr.else = add i32 %ctr.loop.inner, 1
  call void @foo(i32 3)
  %cond.else = icmp slt i32 %ctr.else, %count
  br i1 %cond.else, label %loop.outer, label %exit

; CHECK: Flow2:
; CHECK:   %6 = phi i1 [ %4, %else ], [ true, %Flow1 ]
; CHECK:   br i1 %6, label %exit, label %loop.outer

exit:
  ret void
}

declare void @foo(i32)

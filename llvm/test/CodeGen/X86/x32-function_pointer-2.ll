; RUN: llc < %s -mtriple=x86_64-linux-gnux32 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnux32 -fast-isel | FileCheck %s

; Test call function pointer with function argument
;
; void bar (ptr h, void (*foo) (ptr))
;    {
;      foo (h);
;      foo (h);
;    }


define void @bar(ptr %h, ptr nocapture %foo) nounwind {
entry:
  tail call void %foo(ptr %h) nounwind
; CHECK: mov{{l|q}}	%{{e|r}}si,
; CHECK: callq	*%r
  tail call void %foo(ptr %h) nounwind
; CHECK: jmpq	*%r
  ret void
}

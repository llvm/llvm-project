; RUN: llc < %s -mtriple=x86_64-linux-gnux32 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnux32 -fast-isel | FileCheck %s

; Test call function pointer with function argument
;
; void bar (void * h, void (*foo) (void *))
;    {
;      foo (h);
;      foo (h);
;    }


define void @bar(i8* %h, void (i8*)* nocapture %foo) nounwind {
entry:
  tail call void %foo(i8* %h) nounwind
; CHECK: mov{{l|q}}	%{{e|r}}si,
; CHECK: callq	*%r
  tail call void %foo(i8* %h) nounwind
; CHECK: jmpq	*%r
  ret void
}

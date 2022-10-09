; RUN: opt < %s -aa-pipeline=basic-aa -gvn -S | FileCheck %s

; The input *.ll is obtained by manually annotating "invariant.load" to the 
; two loads. With "invariant.load" metadata, the second load is redundant.
;
; int foo(int *p, char *q) {
;     *q = (char)*p;
;     return *p + 1;
; }

define i32 @foo(i32* nocapture %p, i8* nocapture %q) {
entry:
  %0 = load i32, i32* %p, align 4, !invariant.load !3
  %conv = trunc i32 %0 to i8
  store i8 %conv, i8* %q, align 1
  %1 = load i32, i32* %p, align 4, !invariant.load !3
  %add = add nsw i32 %1, 1
  ret i32 %add

; CHECK: foo
; CHECK: %0 = load i32, i32* %p
; CHECK: store i8 %conv, i8* %q,
; CHECK: %add = add nsw i32 %0, 1
}

!3 = !{}

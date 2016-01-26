; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux < %s | FileCheck %s -check-prefix=CHECK

; Test if the negation of the non-equality check between floating points are
; translated to jnp followed by jne.

; CHECK: jne
; CHECK-NEXT: jnp
define void @foo(float %f) {
entry:
  %cmp = fcmp une float %f, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @a()
  br label %if.end

if.end:
  ret void
}

declare void @a()

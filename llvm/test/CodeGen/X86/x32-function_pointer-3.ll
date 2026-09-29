; RUN: llc < %s -mtriple=x86_64-linux-gnux32 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnux32 -fast-isel | FileCheck %s

; Test calling function pointer passed in struct

;    The function argument `h' in

;    struct foo {
;      void (*f) (void);
;      int i;
;    };
;    void
;    bar (struct foo h)
;    {
;      h.f ();
;    }

;    is passed in the 64-bit %rdi register.  The `f' field is in the lower 32
;    bits of %rdi register and the `i' field is in the upper 32 bits of %rdi
;    register.  We need to zero-extend %edi to %rdi before branching via %rdi.

define void @bar(i64 %h.coerce) nounwind {
entry:
  %h.sroa.0.0.extract.trunc = trunc i64 %h.coerce to i32
  %0 = inttoptr i32 %h.sroa.0.0.extract.trunc to ptr
; CHECK: movl	%edi, %e[[REG:.*]]
  tail call void %0() nounwind
; CHECK: jmpq	*%r[[REG]]
  ret void
}

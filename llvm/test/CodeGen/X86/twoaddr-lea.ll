;; X's live range extends beyond the shift, so the register allocator
;; cannot coalesce it with Y.  Because of this, a copy needs to be
;; emitted before the shift to save the register value before it is
;; clobbered.  However, this copy is not needed if the register
;; allocator turns the shift into an LEA.  This also occurs for ADD.

; Check that the shift gets turned into an LEA.
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-apple-darwin | FileCheck %s

@G = external global i32

define i32 @test1(i32 %X) nounwind {
; CHECK: test1:
; CHECK-NOT: mov
; CHECK: leal 1(%rdi)
        %Z = add i32 %X, 1
        store volatile i32 %Z, i32* @G
        ret i32 %X
}

; rdar://8977508
; The second add should not be transformed to leal nor should it be
; commutted (which would require inserting a copy).
define i32 @test2(i32 inreg %a, i32 inreg %b, i32 %c, i32 %d) nounwind {
entry:
; CHECK: test2:
; CHECK: leal
; CHECK-NOT: leal
; CHECK-NOT: mov
; CHECK-NEXT: addl
; CHECK-NEXT: ret
 %add = add i32 %b, %a
 %add3 = add i32 %add, %c
 %add5 = add i32 %add3, %d
 ret i32 %add5
}

; rdar://9002648
define i64 @test3(i64 %x) nounwind readnone ssp {
entry:
; CHECK: test3:
; CHECK: leaq (%rdi,%rdi), %rax
; CHECK-NOT: addq
; CHECK-NEXT: ret
  %0 = shl i64 %x, 1
  ret i64 %0
}

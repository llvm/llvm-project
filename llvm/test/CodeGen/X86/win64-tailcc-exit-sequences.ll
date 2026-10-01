; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; Exits whose RSP adjustment is not simply the frame size are arranged so that
; the Windows unwinder, which matches epilogues by their instructions, is right
; at every instruction: callee-saved registers are restored with movs (body
; code), and RSP is changed once, by an add that is the first instruction of a
; recognised epilogue. The frame here has a 48 byte reserve for tail calls.

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare tailcc void @h(i64, i64, i64, i64, i64, i64, i64)

define tailcc void @exits(i64 %a, i64 %b, i32 %c) {
; CHECK-LABEL: exits:
; CHECK:         subq $48, %rsp
; CHECK-NEXT:    .seh_stackalloc 48
; CHECK-NEXT:    .seh_endprologue
;
; The largest tail call needs no adjustment, and the store that overwrites the
; return address is the last instruction before the epilogue.
; CHECK:         movq %rdx, 48(%rsp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp g # TAILCALL
;
; A smaller tail call moves RSP up to its return address in one add.
; CHECK:         movq 48(%rsp), %rax
; CHECK-NEXT:    movq %rax, 32(%rsp)
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    addq $32, %rsp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp h # TAILCALL
;
; A return gives the reserve back the same way.
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    addq $48, %rsp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    retq $40
  switch i32 %c, label %ret [ i32 0, label %big
                              i32 1, label %small ]
big:
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
small:
  musttail call tailcc void @h(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %a, i64 %a)
  ret void
ret:
  ret void
}

; A tail call that shrinks the stack argument area needs the same treatment
; even though the function has no reserve and no frame.
define tailcc void @shrink(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i64 %i, i64 %j, i64 %k, i64 %l) {
; CHECK-LABEL: shrink:
; CHECK:         movq (%rsp), %r10
; CHECK-NEXT:    movq %r10, 48(%rsp)
; CHECK:         addq $48, %rsp
; CHECK-NEXT:    jmp h # TAILCALL
  musttail call tailcc void @h(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g)
  ret void
}

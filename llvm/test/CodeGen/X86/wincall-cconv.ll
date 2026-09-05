; RUN: llc -mtriple=x86_64apx-unknown-windows-msvc < %s | FileCheck %s

; The wincall calling convention (x86_64 APX, default for x86_64apx-windows)
; passes the first 8 integer args in RCX,RDX,R8,R9,R16,R17,R18,R19 and the
; first 8 FP/vector args in XMM0-XMM7, allocated independently (no Win64-style
; register pairing/skipping).

declare x86_wincallcc void @wincall_thunk(i64, i64, i64, i64, i64, i64, i64, i64)

; CHECK-LABEL: call_8_int:
; CHECK:       subq $56, %rsp
; CHECK-NEXT:  movq 96(%rsp), %r16
; CHECK-NEXT:  movq 104(%rsp), %r17
; CHECK-NEXT:  movq 112(%rsp), %r18
; CHECK-NEXT:  movq 120(%rsp), %r19
; CHECK-NEXT:  callq wincall_thunk
; CHECK-NEXT:  addq $56, %rsp
; CHECK-NEXT:  retq
define void @call_8_int(i64 %a, i64 %b, i64 %c, i64 %d,
                        i64 %e, i64 %f, i64 %g, i64 %h) nounwind {
entry:
  call x86_wincallcc void @wincall_thunk(i64 %a, i64 %b, i64 %c, i64 %d,
                                         i64 %e, i64 %f, i64 %g, i64 %h)
  ret void
}

; Callee side: the first 8 integer args arrive in RCX,RDX,R8,R9,R16,R17,R18,R19.
; CHECK-LABEL: sum8:
; CHECK:       addq %rdx, %rcx
; CHECK-NEXT:  addq %r9, %r8
; CHECK-NEXT:  addq %r8, %rcx
; CHECK-NEXT:  addq %r17, %r16
; CHECK-NEXT:  addq %r18, %r16
; CHECK-NEXT:  addq %r16, %rcx
; CHECK-NEXT:  addq %r19, %rax
; CHECK-NEXT:  retq
define x86_wincallcc i64 @sum8(i64 %a, i64 %b, i64 %c, i64 %d,
                               i64 %e, i64 %f, i64 %g, i64 %h) nounwind {
entry:
  %s1 = add i64 %a, %b
  %s2 = add i64 %s1, %c
  %s3 = add i64 %s2, %d
  %s4 = add i64 %s3, %e
  %s5 = add i64 %s4, %f
  %s6 = add i64 %s5, %g
  %s7 = add i64 %s6, %h
  ret i64 %s7
}

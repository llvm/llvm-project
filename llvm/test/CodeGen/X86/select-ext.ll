; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

; TODO: (zext(select c, load1, load2)) -> (select c, zextload1, zextload2)

; CHECK-LABEL: foo
; CHECK:        movzbl  (%rdi), %eax
; CHECK-NEXT:   movzbl  1(%rdi), %ecx
; CHECK-NEXT:   testl   %esi, %esi
; CHECK-NEXT:   cmovel  %eax, %ecx
; CHECK-NEXT:   movzbl  %cl, %eax
; CHECK-NEXT:   retq

define i64 @foo(i8* %p, i1 zeroext %c) {
  %ld1 = load volatile i8, i8* %p
  %arrayidx1 = getelementptr inbounds i8, i8* %p, i64 1
  %ld2 = load volatile i8, i8* %arrayidx1
  %cond.v = select i1 %c, i8 %ld2, i8 %ld1
  %cond = zext i8 %cond.v to i64
  ret i64 %cond
}


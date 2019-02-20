; RUN: llc < %s -mtriple=i686--        | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -O0 | FileCheck %s -check-prefix=CHECK-X64
; RUN: llc < %s -mtriple=x86_64-- -O2 | FileCheck %s -check-prefix=CHECK-X64

; CHECK-LABEL: shift1
define void @shift1(i256 %x, i256 %a, i256* nocapture %r) nounwind readnone {
entry:
	%0 = ashr i256 %x, %a
	store i256 %0, i256* %r
        ret void
}

; CHECK-LABEL: shift2
define i256 @shift2(i256 %c) nounwind
{
  %b = shl i256 1, %c  ; %c must not be a constant
  ; Special case when %c is 0:
  ; CHECK-X64: testb [[REG:%(bpl|r[0-9]+b)]], {{%(bpl|r[0-9]+b)}}
  ; CHECK-X64: cmoveq
  ret i256 %b
}

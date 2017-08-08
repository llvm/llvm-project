; RUN: llc < %s -mtriple=i686-unknown-linux-gnu   | FileCheck %s -check-prefix=X86
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-linux-gnux32      | FileCheck %s -check-prefix=X32

define zeroext i8 @foo() nounwind ssp {
entry:
  %0 = tail call zeroext i16 (...) @bar() nounwind
  %1 = lshr i16 %0, 8
  %2 = trunc i16 %1 to i8
  ret i8 %2

; X86-LABEL: foo
; X86: calll
; X86-NEXT: movb %ah, %al
; X86-NEXT: addl $12, %esp
; X86-NEXT: retl

; X64-LABEL: foo
; X64: callq
; X64-NEXT: # kill
; X64-NEXT: shrl $8, %eax
; X64-NEXT: # kill
; X64-NEXT: popq
; X64-NEXT: retq

; X32-LABEL: foo
; X32: callq
; X32-NEXT: # kill
; X32-NEXT: shrl $8, %eax
; X32-NEXT: # kill
; X32-NEXT: popq
; X32-NEXT: retq
}

declare zeroext i16 @bar(...)

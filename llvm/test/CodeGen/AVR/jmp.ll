; RUN: llc -filetype=obj -mtriple=avr < %s | llvm-objdump -dr --no-show-raw-insn - | FileCheck %s

define i8 @foo(i8 %a) {
bb0:
  %0 = tail call i8 @bar(i8 %a)
  %1 = icmp eq i8 %0, 123
  br i1 %1, label %bb1, label %bb2

bb1:
  ret i8 100

bb2:
  ret i8 200
}

declare i8 @bar(i8);

; CHECK: rcall   .-2
; CHECK-NEXT: 00000000: R_AVR_13_PCREL bar
; CHECK-NEXT: cpi     r24, 0x7b
; CHECK-NEXT: brne    .+4
; CHECK-NEXT: ldi     r24, 0x64
; CHECK-NEXT: ret
; CHECK-NEXT: ldi     r24, 0xc8
; CHECK-NEXT: ret

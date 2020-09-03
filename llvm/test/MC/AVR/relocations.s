; RUN: llvm-mc -filetype=obj -triple=avr %s -mattr=avr6 | llvm-objdump -r - | FileCheck %s

; CHECK: RELOCATION RECORDS FOR

.global bar
bar:
  jmp bar

; CHECK: R_AVR_LDI SYMBOL+0x3
ldi r21, SYMBOL+3

; CHECK: R_AVR_6 SYMBOL+0x4
ldd r8, Y+SYMBOL+4

; CHECK-NEXT: R_AVR_6_ADIW FOO
adiw r24, FOO

; CHECK-NEXT: R_AVR_16 BAR+0x2
lds r8, BAR+2

; CHECK-NEXT: R_AVR_PORT5 foo
sbic foo, 1

; CHECK-NEXT: R_AVR_PORT6 Bar+0x2
in r30, Bar+2

; CHECK-NEXT: R_AVR_CALL foo+0x3
call foo+3

; CHECK-NEXT: R_AVR_13_PCREL foo-0xa
rjmp foo-10

; CHECK-NEXT: R_AVR_7_PCREL foo+0x2
brcs foo+2

; CHECK-NEXT: R_AVR_LO8_LDI bar+0x3
; CHECK-NEXT: R_AVR_LO8_LDI abc
; CHECK-NEXT: R_AVR_LO8_LDI abc
ldi r24, lo8(bar+3)
ldi r16, +lo8(abc)
ldi r16, lo8(+(abc))

; CHECK-NEXT: R_AVR_HI8_LDI abc
; CHECK-NEXT: R_AVR_HI8_LDI abc
; CHECK-NEXT: R_AVR_HI8_LDI abc
ldi r30, hi8(abc)
ldi r16, +hi8(abc)
ldi r16, hi8(+(abc))

; CHECK-NEXT: R_AVR_LO8_LDI_NEG abc
; CHECK-NEXT: R_AVR_LO8_LDI_NEG abc
ldi r16, -lo8(abc)
ldi r16, lo8(-(abc))

; CHECK-NEXT: R_AVR_HI8_LDI_NEG abc
; CHECK-NEXT: R_AVR_HI8_LDI_NEG abc
ldi r16, -hi8(abc)
ldi r16, hi8(-(abc))

; CHECK-NEXT: R_AVR_HH8_LDI foo
; CHECK-NEXT: R_AVR_HH8_LDI foo
; CHECK-NEXT: R_AVR_HH8_LDI foo
ldi r16, hh8(foo)
ldi r16, +hh8(foo)
ldi r16, hh8(+(foo))

; CHECK-NEXT: R_AVR_HH8_LDI_NEG foo
; CHECK-NEXT: R_AVR_HH8_LDI_NEG foo
ldi r16, -hh8(foo)
ldi r16, hh8(-(foo))

; CHECK-NEXT: R_AVR_HH8_LDI foo
; CHECK-NEXT: R_AVR_HH8_LDI foo
; CHECK-NEXT: R_AVR_HH8_LDI foo
ldi r24, hlo8(foo)
ldi r24, +hlo8(foo)
ldi r24, hlo8(+(foo))

; CHECK-NEXT: R_AVR_HH8_LDI_NEG foo
; CHECK-NEXT: R_AVR_HH8_LDI_NEG foo
ldi r24, -hlo8(foo)
ldi r24, hlo8(-(foo))

; CHECK-NEXT: R_AVR_MS8_LDI bar
; CHECK-NEXT: R_AVR_MS8_LDI bar
; CHECK-NEXT: R_AVR_MS8_LDI bar
ldi r24, hhi8(bar)
ldi r24, +hhi8(bar)
ldi r24, hhi8(+(bar))

; CHECK-NEXT: R_AVR_MS8_LDI_NEG bar
; CHECK-NEXT: R_AVR_MS8_LDI_NEG bar
ldi r24, -hhi8(bar)
ldi r24, hhi8(-(bar))

; CHECK-NEXT: R_AVR_LO8_LDI_PM foo
; CHECK-NEXT: R_AVR_LO8_LDI_PM foo
; CHECK-NEXT: R_AVR_LO8_LDI_PM foo
ldi r17, pm_lo8(foo)
ldi r25, +pm_lo8(foo)
ldi r25, pm_lo8(+(foo))

; CHECK-NEXT: R_AVR_HI8_LDI_PM bar
; CHECK-NEXT: R_AVR_HI8_LDI_PM foo
; CHECK-NEXT: R_AVR_HI8_LDI_PM foo
ldi r22, pm_hi8(bar)
ldi r25, +pm_hi8(foo)
ldi r25, pm_hi8(+(foo))

; CHECK-NEXT: R_AVR_HH8_LDI_PM baz
; CHECK-NEXT: R_AVR_HH8_LDI_PM foo
; CHECK-NEXT: R_AVR_HH8_LDI_PM foo
ldi r25, pm_hh8(baz)
ldi r25, +pm_hh8(foo)
ldi r25, pm_hh8(+(foo))

; CHECK-NEXT: R_AVR_LO8_LDI_PM_NEG foo
; CHECK-NEXT: R_AVR_LO8_LDI_PM_NEG foo
ldi r25, -pm_lo8(foo)
ldi r25, pm_lo8(-(foo))

; CHECK-NEXT: R_AVR_HI8_LDI_PM_NEG foo
; CHECK-NEXT: R_AVR_HI8_LDI_PM_NEG foo
ldi r25, -pm_hi8(foo)
ldi r25, pm_hi8(-(foo))

; CHECK-NEXT: R_AVR_HH8_LDI_PM_NEG foo
; CHECK-NEXT: R_AVR_HH8_LDI_PM_NEG foo
ldi r25, -pm_hh8(foo)
ldi r25, pm_hh8(-(foo))

; CHECK-NEXT: R_AVR_LO8_LDI_GS foo
ldi r17, lo8(gs(foo))

; CHECK-NEXT: R_AVR_HI8_LDI_GS foo
ldi r18, hi8(gs(foo))

; CHECK-NEXT: R_AVR_16
.short foo

; CHECK-NEXT: R_AVR_16_PM
.short gs(foo)

; CHECK-NEXT: R_AVR_8
.byte foo

; CHECK-NEXT: R_AVR_8_LO8
.byte lo8(foo)

; CHECK-NEXT: R_AVR_8_HI8
.byte hi8(foo)

; CHECK-NEXT: R_AVR_8_HLO8
.byte hlo8(foo)

; CHECK: R_AVR_DIFF8
.byte foo - bar

; CHECK: R_AVR_DIFF16
.short foo - bar

; CHECK: R_AVR_DIFF32
.long foo - bar

; RUN: llc -mattr=avr6,sram < %s -march=avr | FileCheck %s

;TODO: test returning byval structs
; TODO: test naked functions

define void @return_void() {
; CHECK: return_void:{{[a-zA-Z0-9 #@]*}}
; CHECK-NEXT: {{.*}}:
; CHECK-NEXT: ret
    ret void
}

define i8 @return8_imm() {
; CHECK-LABEL: return8_imm:
; CHECK: ldi r24, 5
    ret i8 5
}

define i8 @return8_arg(i8 %x) {
; CHECK: return8_arg:{{[a-zA-Z0-9 #@]*}}
; CHECK-NEXT: {{.*}}:
; CHECK-NEXT: ret
    ret i8 %x
}

define i8 @return8_arg2(i8 %x, i8 %y, i8 %z) {
; CHECK-LABEL: return8_arg2:
; CHECK: mov r24, r20
    ret i8 %z
}

define i16 @return16_imm() {
; CHECK-LABEL: return16_imm:
; CHECK: ldi r24, 57
; CHECK: ldi r25, 48
    ret i16 12345
}

define i16 @return16_arg(i16 %x) {
; CHECK: return16_arg:{{[a-zA-Z0-9 #@]*}}
; CHECK-NEXT: {{.*}}:
; CHECK-NEXT: ret
    ret i16 %x
}

define i16 @return16_arg2(i16 %x, i16 %y, i16 %z) {
; CHECK-LABEL: return16_arg2:
; CHECK: movw r24, r20
    ret i16 %z
}

define i32 @return32_imm() {
; CHECK-LABEL: return32_imm:
; CHECK: ldi r22, 21
; CHECK: ldi r23, 205
; CHECK: ldi r24, 91
; CHECK: ldi r25, 7
    ret i32 123456789
}

define i32 @return32_arg(i32 %x) {
; CHECK: return32_arg:{{[a-zA-Z0-9 #@]*}}
; CHECK-NEXT: {{.*}}:
; CHECK-NEXT: ret
    ret i32 %x
}

define i32 @return32_arg2(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: return32_arg2:
; CHECK: movw r22, r14
; CHECK: movw r24, r16
    ret i32 %z
}

define i64 @return64_imm() {
; CHECK-LABEL: return64_imm:
; CHECK: ldi r18, 204
; CHECK: ldi r19, 204
; CHECK: ldi r20, 104
; CHECK: ldi r21, 37
; CHECK: ldi r22, 25
; CHECK: ldi r23, 22
; CHECK: ldi r24, 236
; CHECK: ldi r25, 190
    ret i64 13757395258967641292
}

define i64 @return64_arg(i64 %x) {
; CHECK: return64_arg:{{[a-zA-Z0-9 #@]*}}
; CHECK-NEXT: {{.*}}:
; CHECK-NEXT: ret
    ret i64 %x
}

define i64 @return64_arg2(i64 %x, i64 %y, i64 %z) {
; CHECK-LABEL: return64_arg2:
; CHECK: push r28
; CHECK: push r29
; CHECK: ldd r18, Y+3
; CHECK: ldd r19, Y+4
; CHECK: ldd r20, Y+5
; CHECK: ldd r21, Y+6
; CHECK: ldd r22, Y+7
; CHECK: ldd r23, Y+8
; CHECK: ldd r24, Y+9
; CHECK: ldd r25, Y+10
; CHECK: pop r29
; CHECK: pop r28
    ret i64 %z
}

define i32 @return64_trunc(i32 %a, i32 %b, i32 %c, i64 %d) {
; CHECK-LABEL: return64_trunc:
; CHECK: push r28
; CHECK: push r29
; CHECK: ldd r22, Y+3
; CHECK: ldd r23, Y+4
; CHECK: ldd r24, Y+5
; CHECK: ldd r25, Y+6
; CHECK: pop r29
; CHECK: pop r28
  %result = trunc i64 %d to i32
  ret i32 %result
}

define i32 @naked(i32 %x) naked {
; CHECK-LABEL: naked:
; CHECK-NOT: ret
  ret i32 %x
}

define avr_intrcc void @interrupt_handler() {
; CHECK-LABEL: interrupt_handler:
; CHECK: reti
  ret void
}

define avr_signalcc void @signal_handler() {
; CHECK-LABEL: signal_handler:
; CHECK: reti
  ret void
}

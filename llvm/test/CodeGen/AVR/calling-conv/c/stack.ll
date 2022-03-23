; RUN: llc -mtriple=avr < %s | FileCheck %s

; CHECK-LABEL: ret_void_args_i64_i64_i32
define void @ret_void_args_i64_i64_i32(i64 %a, i64 %b, i32 %c) {
  ; We're goign to clobber PTRREG Y
  ; CHECK:      push    r28
  ; CHECK-NEXT: push    r29

  ; Load the stack pointer into Y.
  ; CHECK-NEXT: in      r28, 61
  ; CHECK-NEXT: in      r29, 62

  ; Load the top two bytes from the 32-bit int.
  ; CHECK-NEXT: ldd     r24, Y+7
  ; CHECK-NEXT: ldd     r25, Y+8
  ; Store the top two bytes of the 32-bit int to memory.
  ; CHECK-NEXT: sts     7, r25
  ; CHECK-NEXT: sts     6, r24

  ; Load the bottom two bytes from the 32-bit int.
  ; CHECK-NEXT: ldd     r24, Y+5
  ; CHECK-NEXT: ldd     r25, Y+6
  ; Store the bottom two bytes of the 32-bit int to memory.
  ; CHECK-NEXT: sts     5, r25
  ; CHECK-NEXT: sts     4, r24

  ; Restore PTRREG Y
  ; CHECK-NEXT: pop     r29
  ; CHECK-NEXT: pop     r28
  store volatile i32 %c, i32* inttoptr (i64 4 to i32*)
  ret void
}

; NOTE: All arguments are passed via the stack for varargs functions.
; NOTE: Both %a & %b occupy a 1-byte stack slot.
define i8 @foo0(i8 %a, i8 %b, ...) {
; CHECK-LABEL: foo0:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    push r28
; CHECK-NEXT:    push r29
; CHECK-NEXT:    in r28, 61
; CHECK-NEXT:    in r29, 62
; CHECK-NEXT:    ldd r25, Y+6
; CHECK-NEXT:    ldd r24, Y+5
; CHECK-NEXT:    sub r24, r25
; CHECK-NEXT:    pop r29
; CHECK-NEXT:    pop r28
; CHECK-NEXT:    ret
  %c = sub i8 %a, %b
  ret i8 %c
}

; NOTE: All arguments are passed via the stack since the argument %a is too large.
define i8 @foo1([19 x i8] %a, i8 %b) {
; CHECK-LABEL: foo1:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    push r28
; CHECK-NEXT:    push r29
; CHECK-NEXT:    in r28, 61
; CHECK-NEXT:    in r29, 62
; CHECK-NEXT:    ldd r25, Y+24
; CHECK-NEXT:    ldd r24, Y+5
; CHECK-NEXT:    sub r24, r25
; CHECK-NEXT:    pop r29
; CHECK-NEXT:    pop r28
; CHECK-NEXT:    ret
  %c = extractvalue [19 x i8] %a, 0
  %d = sub i8 %c, %b
  ret i8 %d
}

; NOTE: The argument %b is passed via the stack, since the argument %a costs
; NOTE: total 18 registers though it is a 17-byte array.
define i8 @foo2([17 x i8] %a, i8 %b) {
; CHECK-LABEL: foo2:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    push r8
; CHECK-NEXT:    push r28
; CHECK-NEXT:    push r29
; CHECK-NEXT:    in r28, 61
; CHECK-NEXT:    in r29, 62
; CHECK-NEXT:    ldd r24, Y+6
; CHECK-NEXT:    sub r8, r24
; CHECK-NEXT:    mov r24, r8
; CHECK-NEXT:    pop r29
; CHECK-NEXT:    pop r28
; CHECK-NEXT:    pop r8
; CHECK-NEXT:    ret
  %c = extractvalue [17 x i8] %a, 0
  %d = sub i8 %c, %b
  ret i8 %d
}

; NOTE: Though %a costs 16 registers and 2 registers are vacant, the 4-byte
; NOTE: %b has to be dropped to the stack.
; NOTE: total 18 registers.
define i32 @foo3([4 x i32] %a, i32 %b) {
; CHECK-LABEL: foo3:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    push r28
; CHECK-NEXT:    push r29
; CHECK-NEXT:    in r28, 61
; CHECK-NEXT:    in r29, 62
; CHECK-NEXT:    ldd r22, Y+5
; CHECK-NEXT:    ldd r23, Y+6
; CHECK-NEXT:    ldd r24, Y+7
; CHECK-NEXT:    ldd r25, Y+8
; CHECK-NEXT:    sub r22, r10
; CHECK-NEXT:    sbc r23, r11
; CHECK-NEXT:    sbc r24, r12
; CHECK-NEXT:    sbc r25, r13
; CHECK-NEXT:    pop r29
; CHECK-NEXT:    pop r28
; CHECK-NEXT:    ret
  %c = extractvalue [4 x i32] %a, 0
  %d = sub nsw i32 %b, %c
  ret i32 %d
}

; NOTE: Both %1 and %2 are passed via stack, and each has a 1-byte slot.
define i8 @foo4([17 x i8] %0, i8 %1, i8 %2) {
; CHECK-LABEL: foo4:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    push r8
; CHECK-NEXT:    push r28
; CHECK-NEXT:    push r29
; CHECK-NEXT:    in r28, 61
; CHECK-NEXT:    in r29, 62
; CHECK-NEXT:    ldd r24, Y+6
; CHECK-NEXT:    sub r8, r24
; CHECK-NEXT:    ldd r24, Y+7
; CHECK-NEXT:    add r24, r8
; CHECK-NEXT:    pop r29
; CHECK-NEXT:    pop r28
; CHECK-NEXT:    pop r8
; CHECK-NEXT:    ret
  %4 = extractvalue [17 x i8] %0, 0
  %5 = sub i8 %4, %1
  %6 = add i8 %5, %2
  ret i8 %6
}

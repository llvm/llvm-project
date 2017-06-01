; RUN: llc < %s -march=avr | FileCheck %s

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

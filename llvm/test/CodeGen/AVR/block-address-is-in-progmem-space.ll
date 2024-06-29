; RUN: llc -mcpu=atmega328 < %s -march=avr | FileCheck %s

; This test verifies that the pointer to a basic block
; should always be a pointer in address space 1.
;
; If this were not the case, then programs targeting
; AVR that attempted to read their own machine code
; via a pointer to a label would actually read from RAM
; using a pointer relative to the the start of program flash.
;
; This would cause a load of uninitialized memory, not even
; touching the program's machine code as otherwise desired.

target datalayout = "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8-a:8"

; CHECK-LABEL: load_with_no_forward_reference
define i8 @load_with_no_forward_reference(i8 %a, i8 %b) {
second:
  ; CHECK:      ldi r30, .Ltmp0+2
  ; CHECK-NEXT: ldi r31, .Ltmp0+4
  ; CHECK: lpm r24, Z
  %bar = load i8, ptr addrspace(1) blockaddress(@function_with_no_forward_reference, %second)
  ret i8 %bar
}

; CHECK-LABEL: load_from_local_label
define i8 @load_from_local_label(i8 %a, i8 %b) {
entry:
  %result1 = add i8 %a, %b

  br label %second

; CHECK-LABEL: .Ltmp1:
second:
  ; CHECK:      ldi r30, .Ltmp1+2
  ; CHECK-NEXT: ldi r31, .Ltmp1+4
  ; CHECK-NEXT: lpm r24, Z
  %result2 = load i8, ptr addrspace(1) blockaddress(@load_from_local_label, %second)
  ret i8 %result2
}

; A function with no forward reference, right at the end
; of the file.
define i8 @function_with_no_forward_reference(i8 %a, i8 %b) {
entry:
  %result = add i8 %a, %b
  br label %second
second:
  ret i8 0
}


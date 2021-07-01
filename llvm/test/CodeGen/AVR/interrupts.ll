; RUN: llc < %s -march=avr | FileCheck %s

define avr_intrcc void @interrupt_handler() {
; CHECK-LABEL: interrupt_handler:
; CHECK: sei
; CHECK-NEXT: push r0
; CHECK-NEXT: push r1
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: push r0
; CHECK: clr r0
; CHECK-NEXT: clr r1
; CHECK: pop r0
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: pop r1
; CHECK-NEXT: pop r0
; CHECK-NEXT: reti
  ret void
}

define void @interrupt_handler_via_ir_attribute() #0 {
; CHECK-LABEL: interrupt_handler_via_ir_attribute:
; CHECK: sei
; CHECK-NEXT: push r0
; CHECK-NEXT: push r1
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: push r0
; CHECK: clr r0
; CHECK-NEXT: clr r1
; CHECK: pop r0
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: pop r1
; CHECK-NEXT: pop r0
; CHECK-NEXT: reti
  ret void
}

define avr_signalcc void @signal_handler() {
; CHECK-LABEL: signal_handler:
; CHECK-NOT: sei
; CHECK: push r0
; CHECK-NEXT: push r1
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: push r0
; CHECK: clr r0
; CHECK-NEXT: clr r1
; CHECK: pop r0
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: pop r1
; CHECK-NEXT: pop r0
; CHECK-NEXT: reti
  ret void
}

define void @signal_handler_via_attribute() #1 {
; CHECK-LABEL: signal_handler_via_attribute:
; CHECK-NOT: sei
; CHECK: push r0
; CHECK-NEXT: push r1
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: push r0
; CHECK: clr r0
; CHECK-NEXT: clr r1
; CHECK: pop r0
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: pop r1
; CHECK-NEXT: pop r0
; CHECK-NEXT: reti
  ret void
}

define avr_intrcc void @interrupt_alloca() {
; CHECK-LABEL: interrupt_alloca:
; CHECK: sei
; CHECK-NEXT: push r0
; CHECK-NEXT: push r1
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: push r0
; CHECK: clr r0
; CHECK-NEXT: clr r1
; CHECK: push r28
; CHECK-NEXT: push r29
; CHECK-NEXT: in r28, 61
; CHECK-NEXT: in r29, 62
; CHECK-NEXT: sbiw r28, 1
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
; CHECK: adiw r28, 1
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
; CHECK-NEXT: pop r29
; CHECK-NEXT: pop r28
; CHECK: pop r0
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: pop r1
; CHECK-NEXT: pop r0
; CHECK-NEXT: reti
  alloca i8
  ret void
}

attributes #0 = { "interrupt" }
attributes #1 = { "signal" }

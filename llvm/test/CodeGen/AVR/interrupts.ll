; RUN: llc < %s -march=avr | FileCheck %s

@count = global i8 0

define avr_intrcc void @interrupt_handler() {
; CHECK-LABEL: interrupt_handler:
; CHECK: sei
; CHECK-NEXT: push r0
; CHECK-NEXT: push r1
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: push r0
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

define void @signal_handler_with_increment() #1 {
; CHECK-LABEL: signal_handler_with_increment:
; CHECK:      push r0
; CHECK-NEXT: push r1
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: push r0
; CHECK-NEXT: clr r1
; CHECK-NEXT: push r24
; CHECK-NEXT: lds r24, count
; CHECK-NEXT: inc r24
; CHECK-NEXT: sts count, r24
; CHECK-NEXT: pop r24
; CHECK-NEXT: pop r0
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: pop r1
; CHECK-NEXT: pop r0
; CHECK-NEXT: reti
  %old = load volatile i8, i8* @count
  %new = add i8 %old, 1
  store volatile i8 %new, i8* @count
  ret void
}

declare void @foo()

; When a signal handler calls a function, it must push/pop all call clobbered
; registers.
define void @signal_handler_with_call() #1 {
; CHECK-LABEL: signal_handler_with_call:
; CHECK:      push r0
; CHECK-NEXT: push r1
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: push r0
; CHECK-NEXT: clr r1
; CHECK-NEXT: push r18
; CHECK-NEXT: push r19
; CHECK-NEXT: push r20
; CHECK-NEXT: push r21
; CHECK-NEXT: push r22
; CHECK-NEXT: push r23
; CHECK-NEXT: push r24
; CHECK-NEXT: push r25
; CHECK-NEXT: push r26
; CHECK-NEXT: push r27
; CHECK-NEXT: push r30
; CHECK-NEXT: push r31
; CHECK-NEXT: call foo
; CHECK-NEXT: pop r31
; CHECK-NEXT: pop r30
; CHECK-NEXT: pop r27
; CHECK-NEXT: pop r26
; CHECK-NEXT: pop r25
; CHECK-NEXT: pop r24
; CHECK-NEXT: pop r23
; CHECK-NEXT: pop r22
; CHECK-NEXT: pop r21
; CHECK-NEXT: pop r20
; CHECK-NEXT: pop r19
; CHECK-NEXT: pop r18
; CHECK-NEXT: pop r0
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: pop r1
; CHECK-NEXT: pop r0
; CHECK-NEXT: reti
  call void @foo()
  ret void
}

attributes #0 = { "interrupt" }
attributes #1 = { "signal" }

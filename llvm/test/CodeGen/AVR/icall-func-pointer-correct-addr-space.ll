; RUN: llc -mattr=lpm,lpmw < %s -march=avr | FileCheck %s

declare void @callback(i16 zeroext)

; CHECK-LABEL: foo
define void @foo() {
entry:
  ; CHECK:      ldi     r{{[0-9]+}}, pm_lo8(callback)
  ; CHECK-NEXT: ldi     r{{[0-9]+}}, pm_hi8(callback)
  call void @bar(i8 zeroext undef, void (i16)* @callback)
  ret void
}

declare void @bar(i8 zeroext, void (i16)*)


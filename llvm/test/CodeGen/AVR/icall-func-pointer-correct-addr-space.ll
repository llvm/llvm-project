; RUN: llc -mattr=lpm,lpmw < %s -march=avr | FileCheck %s

@callbackPtr = common global void (i16)* null, align 8
@myValuePtr = common global i16* null, align 8

@externalConstant = external global i16, align 2

declare void @externalFunction(i16 signext)
declare void @bar(i8 signext, void (i16)*, i16*)

; CHECK-LABEL: loadCallbackPtr
define void @loadCallbackPtr() {
entry:
  ; CHECK:      ldi     r{{[0-9]+}}, pm_lo8(externalFunction)
  ; CHECK-NEXT: ldi     r{{[0-9]+}}, pm_hi8(externalFunction)
  store void (i16)* @externalFunction, void (i16)** @callbackPtr, align 8
  ret void
}

; CHECK-LABEL: loadValuePtr
define void @loadValuePtr() {
entry:
  ; CHECK:      ldi     r{{[0-9]+}}, lo8(externalConstant)
  ; CHECK-NEXT: ldi     r{{[0-9]+}}, hi8(externalConstant)
  store i16* @externalConstant, i16** @myValuePtr, align 8
  ret void
}

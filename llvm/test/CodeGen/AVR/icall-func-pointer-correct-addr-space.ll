; RUN: llc -mattr=lpm,lpmw < %s -march=avr | FileCheck %s

@callbackPtr = common global ptr addrspace(1) null, align 8
@myValuePtr = common global ptr null, align 8

@externalConstant = external global i16, align 2

declare void @externalFunction(i16 signext)
declare void @bar(i8 signext, ptr, ptr)

; CHECK-LABEL: loadCallbackPtr
define void @loadCallbackPtr() {
entry:
  ; CHECK:      ldi     r{{[0-9]+}}, pm_lo8(externalFunction)
  ; CHECK-NEXT: ldi     r{{[0-9]+}}, pm_hi8(externalFunction)
  store ptr addrspace(1) @externalFunction, ptr @callbackPtr, align 8
  ret void
}

; CHECK-LABEL: loadValuePtr
define void @loadValuePtr() {
entry:
  ; CHECK:      ldi     r{{[0-9]+}}, lo8(externalConstant)
  ; CHECK-NEXT: ldi     r{{[0-9]+}}, hi8(externalConstant)
  store ptr @externalConstant, ptr @myValuePtr, align 8
  ret void
}

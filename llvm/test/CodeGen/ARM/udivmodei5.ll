; RUN: llc -mtriple=arm-eabi < %s | FileCheck %s

define i65 @udiv65(i65 %a, i65 %b) nounwind {
; CHECK-LABEL: udiv65:
; CHECK-NOT:     call
  %res = udiv i65 %a, %b
  ret i65 %res
}

define i129 @udiv129(i129 %a, i129 %b) nounwind {
; CHECK-LABEL: udiv129:
; CHECK-NOT:     call
  %res = udiv i129 %a, %b
  ret i129 %res
}

define i129 @urem129(i129 %a, i129 %b) nounwind {
; CHECK-LABEL: urem129:
; CHECK-NOT:     call
  %res = urem i129 %a, %b
  ret i129 %res
}

define i129 @sdiv129(i129 %a, i129 %b) nounwind {
; CHECK-LABEL: sdiv129:
; CHECK-NOT:     call
  %res = sdiv i129 %a, %b
  ret i129 %res
}

define i129 @srem129(i129 %a, i129 %b) nounwind {
; CHECK-LABEL: srem129:
; CHECK-NOT:     call
  %res = srem i129 %a, %b
  ret i129 %res
}

; Some higher sizes
define i257 @sdiv257(i257 %a, i257 %b) nounwind {
; CHECK-LABEL: sdiv257:
; CHECK-NOT:     call
  %res = sdiv i257 %a, %b
  ret i257 %res
}

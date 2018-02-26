; RUN: llc < %s -mtriple=i686-unknown-unknown | FileCheck %s

define i32 @shl16sar15(i32 %a) #0 {
; CHECK-LABEL: shl16sar15:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movswl {{[0-9]+}}(%esp), %eax
  %1 = shl i32 %a, 16
  %2 = ashr exact i32 %1, 15
  ret i32 %2
}

define i32 @shl16sar17(i32 %a) #0 {
; CHECK-LABEL: shl16sar17:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movswl {{[0-9]+}}(%esp), %eax
  %1 = shl i32 %a, 16
  %2 = ashr exact i32 %1, 17
  ret i32 %2
}

define i32 @shl24sar23(i32 %a) #0 {
; CHECK-LABEL: shl24sar23:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movsbl {{[0-9]+}}(%esp), %eax
  %1 = shl i32 %a, 24
  %2 = ashr exact i32 %1, 23
  ret i32 %2
}

define i32 @shl24sar25(i32 %a) #0 {
; CHECK-LABEL: shl24sar25:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movsbl {{[0-9]+}}(%esp), %eax
  %1 = shl i32 %a, 24
  %2 = ashr exact i32 %1, 25
  ret i32 %2
}

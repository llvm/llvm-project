; RUN: llc -filetype=asm -o - -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

define i32 @callee() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: .Ltmp0:
; CHECK:              b .Ltmp1
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionEntry
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
; CHECK-LABEL: .Ltmp1:
  ret i32 0
; CHECK-LABEL: .Ltmp2:
; CHECK:              blr
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionExit
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
}

define i32 @caller() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: .Ltmp3:
; CHECK:              b .Ltmp4
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionEntry
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
; CHECK-LABEL: .Ltmp4:
  %retval = tail call i32 @callee()
  ret i32 %retval
; CHECK-LABEL: .Ltmp5:
; CHECK:              blr
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionExit
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
}

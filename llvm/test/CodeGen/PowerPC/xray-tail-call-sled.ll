; RUN: llc -relocation-model=pic -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

define i32 @callee() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: callee:
; CHECK:       .Ltmp[[#l:]]:
; CHECK-NEXT:         b .Ltmp[[#l+1]]
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionEntry
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
; CHECK-NEXT:  .Ltmp[[#l+1]]:
  ret i32 0
; CHECK:       .Ltmp[[#]]:
; CHECK:              blr
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionExit
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
}

define i32 @caller() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: caller:
; CHECK:       .Ltmp[[#l:]]:
; CHECK-NEXT:         b .Ltmp[[#l+1]]
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionEntry
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
; CHECK-NEXT:  .Ltmp[[#l+1]]:
  %retval = tail call i32 @callee()
  ret i32 %retval
; CHECK:       .Ltmp[[#l+2]]:
; CHECK-NEXT:         blr
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionExit
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
}

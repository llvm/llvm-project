;; Test if a potential indirect call target function which has internal linkage and
;; address taken has its type ID emitted to callgraph section.
;; This test also makes sure that callback functions which meet the above constraint
;; are handled correctly.

; RUN: llc -mtriple=arm-unknown-linux --call-graph-section -o - < %s | FileCheck %s

declare !callgraph !0 void @_Z6doWorkPFviE(ptr)

define i32 @_Z4testv() !callgraph !1 {
entry:
  call void @_Z6doWorkPFviE(ptr nonnull @_ZL10myCallbacki)
  ret i32 0
}

; CHECK: _ZL10myCallbacki:
define internal void @_ZL10myCallbacki(i32 %value) !callgraph !2 {
entry:
  %sink = alloca i32, align 4
  store volatile i32 %value, ptr %sink, align 4
  %i1 = load volatile i32, ptr %sink, align 4
  ret void
}

!0 = !{!"_ZTSFvPFviEE"}
!1 = !{!"_ZTSFivE"}
!2 = !{!"_ZTSFviE"}

; CHECK: .section .llvm.callgraph,"o",%llvm_call_graph,.text
;; Version
; CHECK-NEXT: .byte   0
;; Flags -- Potential indirect target so LSB is set to 1. Other bits are 0.
; CHECK-NEXT: .byte   1
;; Function Entry PC
; CHECK-NEXT: .long _ZL10myCallbacki
;; Function type ID -8738933900360652027
; CHECK-NEXT: .long 560098053
; CHECK-NEXT: .long 2260275691

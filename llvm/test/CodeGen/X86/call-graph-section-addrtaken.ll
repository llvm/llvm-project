;; Test if a potential indirect call target function which has internal linkage and
;; address taken has its type ID emitted to callgraph section.
;; This test also makes sure that callback functions which meet the above constraint
;; are handled correctly.

; RUN: llc -mtriple=x86_64-unknown-linux --call-graph-section -o - < %s | FileCheck %s

declare !type !0 void @_Z6doWorkPFviE(ptr)

define i32 @_Z4testv() !type !1 {
entry:
  call void @_Z6doWorkPFviE(ptr nonnull @_ZL10myCallbacki)
  ret i32 0
}

; CHECK: _ZL10myCallbacki:
; CHECK-NEXT: [[LABEL_FUNC:\.Lfunc_begin[0-9]+]]:
define internal void @_ZL10myCallbacki(i32 %value) !type !2 {
entry:
  %sink = alloca i32, align 4
  store volatile i32 %value, ptr %sink, align 4
  %i1 = load volatile i32, ptr %sink, align 4
  ret void
}

!0 = !{i64 0, !"_ZTSFvPFviEE.generalized"}
!1 = !{i64 0, !"_ZTSFivE.generalized"}
!2 = !{i64 0, !"_ZTSFviE.generalized"}

; CHECK: .section .callgraph,"o",@progbits,.text
;; Version
; CHECK-NEXT: .byte   0
;; Flags -- Potential indirect target so LSB is set to 1. Other bits are 0.
; CHECK-NEXT: .byte   1
;; Function Entry PC
; CHECK-NEXT: .quad   [[LABEL_FUNC]]
;; Function type ID
; CHECK-NEXT: .quad   -5212364466660467813

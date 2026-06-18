; Test if a potential indirect call target function which has internal linkage and
; address taken has its type ID emitted to callgraph section.
; This test also makes sure that callback functions which meet the above constraint
; are handled correctly.

; RUN: llc -mtriple=arm-unknown-linux --call-graph-section -o - < %s | FileCheck %s

declare !callgraph !0 void @_Z6doWorkPFviE(ptr)

define i32 @_Z4testv() !callgraph !1 {
entry:
  call void @_Z6doWorkPFviE(ptr nonnull @_ZL10myCallbacki)
  ret i32 0
}

define internal void @_ZL10myCallbacki(i32 %value) !callgraph !2 {
entry:
  %sink = alloca i32, align 4
  store volatile i32 %value, ptr %sink, align 4
  %i1 = load volatile i32, ptr %sink, align 4
  ret void
}

!0 = !{!"_ZTSFvPFviEE.generalized"}
!1 = !{!"_ZTSFivE.generalized"}
!2 = !{!"_ZTSFviE.generalized", i1 true}

; CHECK: .section        .llvm.callgraph,"o",%llvm_call_graph,.text
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .byte   3
; CHECK-NEXT: .long   {{.+}}
; CHECK-NEXT: .long   {{.+}}
; CHECK-NEXT: .long   _Z4testv
; CHECK-NEXT: .long   {{.+}}
; CHECK-NEXT: .long   {{.+}}
; CHECK-NEXT: .byte   1
; CHECK-NEXT: .long   _Z6doWorkPFviE

; CHECK: .section        .llvm.callgraph,"o",%llvm_call_graph,.text
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .byte   9
; CHECK-NEXT: .long   {{.+}}
; CHECK-NEXT: .long   {{.+}}
; CHECK-NEXT: .long   _ZL10myCallbacki
; CHECK-NEXT: .long   {{.+}}
; CHECK-NEXT: .long   {{.+}}

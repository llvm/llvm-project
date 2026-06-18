;; Test if temporary labels are generated for each indirect callsite.
;; Test if the .llvm.callgraph section contains the MD5 hash of callees' type (type id)
;; is correctly paired with its corresponding temporary label generated for indirect
;; call sites annotated with !callee_type metadata.
;; Test if the .llvm.callgraph section contains unique direct callees.

; RUN: llc -mtriple=x86_64-unknown-linux --call-graph-section -o - < %s | FileCheck %s

declare !callgraph !0 void @direct_foo()
declare !callgraph !1 i32 @direct_bar(i8)
declare !callgraph !2 ptr @direct_baz(ptr)

; CHECK: ball:
define ptr @ball() {
entry:
  call void @direct_foo()
  %fp_foo_val = load ptr, ptr null, align 8   
  call void (...) %fp_foo_val(), !callee_type !0   
  call void @direct_foo()
  %fp_bar_val = load ptr, ptr null, align 8  
  %call_fp_bar = call i32 %fp_bar_val(i8 0), !callee_type !2  
  %call_fp_bar_direct = call i32 @direct_bar(i8 1)
  %fp_baz_val = load ptr, ptr null, align 8
  %call_fp_baz = call ptr %fp_baz_val(ptr null), !callee_type !4
  call void @direct_foo()
  %call_fp_baz_direct = call ptr @direct_baz(ptr null)
  call void @direct_foo()
  ret ptr %call_fp_baz
}

!0 = !{!1}
!1 = !{!"_ZTSFvE.generalized"}
!2 = !{!3}
!3 = !{!"_ZTSFicE.generalized"}
!4 = !{!5}
!5 = !{!"_ZTSFPvS_E.generalized"}

; CHECK: .section	.llvm.callgraph,"o",@llvm_call_graph,.text
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .byte   7
; CHECK-NEXT: .quad   5977508082728489289
; CHECK-NEXT: .quad   ball
; CHECK-NEXT: .quad   0
; CHECK-NEXT: .byte   3
; CHECK-NEXT: .quad   direct_foo
; CHECK-NEXT: .quad   direct_bar
; CHECK-NEXT: .quad   direct_baz
; CHECK-NEXT: .byte   3
; CHECK-NEXT: .quad   4524972987496481828
; CHECK-NEXT: .quad   3498816979441845844
; CHECK-NEXT: .quad   8646233951371320954

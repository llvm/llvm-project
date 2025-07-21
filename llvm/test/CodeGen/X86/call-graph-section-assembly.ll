;; Test if temporary labels are generated for each indirect callsite with a callee_type metadata.
;; Test if the .callgraph section contains the numerical callee type id for each of the temporary 
;; labels generated. 

; RUN: llc -mtriple=x86_64-unknown-linux --call-graph-section -o - < %s | FileCheck %s

; CHECK: ball:
; CHECK-NEXT: .Lfunc_begin0:
define ptr @ball() {
entry:
  %fp_foo_val = load ptr, ptr null, align 8
   ; CHECK: .Ltmp0:
  call void (...) %fp_foo_val(), !callee_type !0
  %fp_bar_val = load ptr, ptr null, align 8
  ; CHECK: .Ltmp1:
  %call_fp_bar = call i32 %fp_bar_val(i8 0), !callee_type !2
  %fp_baz_val = load ptr, ptr null, align 8
  ; CHECK: .Ltmp2:
  %call_fp_baz = call ptr %fp_baz_val(ptr null), !callee_type !4
  ret ptr %call_fp_baz
}

; CHECK: .section .callgraph,"o",@progbits,.text

; CHECK-NEXT: .quad   0
; CHECK-NEXT: .quad   .Lfunc_begin0
; CHECK-NEXT: .quad   1
; CHECK-NEXT: .quad   3
; CHECK-NEXT: .quad   4524972987496481828
; CHECK-NEXT: .quad   .Ltmp0
!0 = !{!1}
!1 = !{i64 0, !"_ZTSFvE.generalized"}
; CHECK-NEXT: .quad   3498816979441845844
; CHECK-NEXT: .quad   .Ltmp1
!2 = !{!3}
!3 = !{i64 0, !"_ZTSFicE.generalized"}
; CHECK-NEXT: .quad   8646233951371320954
; CHECK-NEXT: .quad   .Ltmp2
!4 = !{!5}
!5 = !{i64 0, !"_ZTSFPvS_E.generalized"}

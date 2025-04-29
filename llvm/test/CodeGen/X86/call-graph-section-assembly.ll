;; Test if temporary labels are generated for each indirect callsite with a callee_type metadata.
;; Test if the .callgraph section contains the numerical callee type id for each of the temporary 
;; labels generated. 

; RUN: llc -mtriple=x86_64-unknown-linux --call-graph-section -o - < %s | FileCheck %s

declare !type !0 void @foo()

declare !type !1 i32 @bar(i8 signext)

declare !type !2 ptr @baz(ptr)

; CHECK: ball:
; CHECK-NEXT: .Lfunc_begin0:
define void @ball() {
entry:
  %retval = alloca i32, align 4
  %fp_foo = alloca ptr, align 8
  %a = alloca i8, align 1
  %fp_bar = alloca ptr, align 8
  %fp_baz = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store ptr @foo, ptr %fp_foo, align 8
  %fp_foo_val = load ptr, ptr %fp_foo, align 8
  ; CHECK: .Ltmp0:
  call void (...) %fp_foo_val(), !callee_type !1
  store ptr @bar, ptr %fp_bar, align 8
  %fp_bar_val = load ptr, ptr %fp_bar, align 8
  %a_val = load i8, ptr %a, align 1
  ; CHECK: .Ltmp1:
  %call_fp_bar = call i32 %fp_bar_val(i8 signext %a_val), !callee_type !3
  store ptr @baz, ptr %fp_baz, align 8
  %fp_baz_val = load ptr, ptr %fp_baz, align 8
  ; CHECK: .Ltmp2:
  %call_fp_baz = call ptr %fp_baz_val(ptr %a), !callee_type !5
  call void @foo()
  %a_val_2 = load i8, ptr %a, align 1
  %call_bar = call i32 @bar(i8 signext %a_val_2)
  %call_baz = call ptr @baz(ptr %a)
  ret void
}

; CHECK: .section .callgraph,"o",@progbits,.text

; CHECK-NEXT: .quad   0
; CHECK-NEXT: .quad   .Lfunc_begin0
; CHECK-NEXT: .quad   1
; CHECK-NEXT: .quad   3
; CHECK-NEXT: .quad   4524972987496481828
; CHECK-NEXT: .quad   .Ltmp0
!0 = !{i64 0, !"_ZTSFvE.generalized"}
!1 = !{!0}
; CHECK-NEXT: .quad   3498816979441845844
; CHECK-NEXT: .quad   .Ltmp1
!2 = !{i64 0, !"_ZTSFicE.generalized"}
!3 = !{!2}
; CHECK-NEXT: .quad   8646233951371320954
; CHECK-NEXT: .quad   .Ltmp2
!4 = !{i64 0, !"_ZTSFPvS_E.generalized"}
!5 = !{!4}

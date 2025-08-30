;; Test if temporary labels are generated for each callsite.
;; Test if the .callgraph section contains the MD5 hash of callees' type (type id)
;; is correctly paired with its corresponding temporary label generated for indirect
;; call sites annotated with !callee_type metadata.
;; Test if the .callgraph section contains direct callsite temporary labels paired
;; correctly with the corresponding callee symbol.

; RUN: llc -mtriple=x86_64-unknown-linux --call-graph-section -o - < %s | FileCheck %s

declare !type !0 void @direct_foo()
declare !type !1 i32 @direct_bar(i8)
declare !type !2 ptr @direct_baz(ptr)

; CHECK: ball:
; CHECK-NEXT: [[LABEL_FUNC:\.Lfunc_begin[0-9]+]]:
define ptr @ball() {
entry:
  %fp_foo_val = load ptr, ptr null, align 8
   ; CHECK: [[LABEL_TMP0:\.L.*]]:
  call void (...) %fp_foo_val(), !callee_type !0
   ; CHECK: [[LABEL_TMP_DIRECT0:\.L.*]]:
  call void @direct_foo()
  %fp_bar_val = load ptr, ptr null, align 8
  ; CHECK: [[LABEL_TMP1:\.L.*]]:
  %call_fp_bar = call i32 %fp_bar_val(i8 0), !callee_type !2
  ; CHECK: [[LABEL_TMP_DIRECT1:\.L.*]]:
  %call_fp_bar_direct = call i32 @direct_bar(i8 1)
  %fp_baz_val = load ptr, ptr null, align 8
  ; CHECK: [[LABEL_TMP2:\.L.*]]:
  %call_fp_baz = call ptr %fp_baz_val(ptr null), !callee_type !4
  ; CHECK: [[LABEL_TMP_DIRECT2:\.L.*]]:
  %call_fp_baz_direct = call ptr @direct_baz(ptr null)
  ret ptr %call_fp_baz
}

; CHECK: .section .callgraph,"o",@progbits,.text

; CHECK-NEXT: .quad   0
; CHECK-NEXT: .quad   [[LABEL_FUNC]]
; CHECK-NEXT: .quad   1
; CHECK-NEXT: .quad   3
!0 = !{!1}
!1 = !{i64 0, !"_ZTSFvE.generalized"}
;; Test for MD5 hash of _ZTSFvE.generalized and the generated temporary callsite label.
; CHECK-NEXT: .quad   4524972987496481828
; CHECK-NEXT: .quad   [[LABEL_TMP0]]
!2 = !{!3}
!3 = !{i64 0, !"_ZTSFicE.generalized"}
;; Test for MD5 hash of _ZTSFicE.generalized and the generated temporary callsite label.
; CHECK-NEXT: .quad   3498816979441845844
; CHECK-NEXT: .quad   [[LABEL_TMP1]]
!4 = !{!5}
!5 = !{i64 0, !"_ZTSFPvS_E.generalized"}
;; Test for MD5 hash of _ZTSFPvS_E.generalized and the generated temporary callsite label.
; CHECK-NEXT: .quad   8646233951371320954
; CHECK-NEXT: .quad   [[LABEL_TMP2]]
;; Test for number of direct calls and {callsite_label, callee} pairs.
; CHECK-NEXT: .quad	3
; CHECK-NEXT: .quad	[[LABEL_TMP_DIRECT0]]
; CHECK-NEXT: .quad	direct_foo
; CHECK-NEXT: .quad	[[LABEL_TMP_DIRECT1]]
; CHECK-NEXT: .quad	direct_bar
; CHECK-NEXT: .quad	[[LABEL_TMP_DIRECT2]]
; CHECK-NEXT: .quad	direct_baz
